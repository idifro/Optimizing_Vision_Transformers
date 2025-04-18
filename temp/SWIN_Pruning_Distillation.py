import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from einops import rearrange
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import time
from thop import profile

# Ensure multiprocessing starts correctly
mp.set_start_method("spawn", force=True)

# Define Window Partitioning functions
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return x

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H // window_size * W // window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

# Define Swin Transformer Components
class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0)
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkv)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1)
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.proj(out)

class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift=False):
        super().__init__()
        self.window_size = window_size
        self.shift = shift
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.2),  # Increased dropout
            nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(0.2)  # Increased dropout
    
    def forward(self, x, H, W):
        shortcut = x
        x = self.norm1(x).view(-1, H, W, x.shape[-1])
        if self.shift:
            x = torch.roll(x, shifts=(-self.window_size // 2, -self.window_size // 2), dims=(1, 2))
        windows = window_partition(x, self.window_size)
        windows = windows.view(-1, self.window_size * self.window_size, x.shape[-1])
        attn_windows = self.attn(windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, x.shape[-1])
        x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift:
            x = torch.roll(x, shifts=(self.window_size // 2, self.window_size // 2), dims=(1, 2))
        x = x.view(-1, H * W, x.shape[-1])
        x = shortcut + self.drop(x)
        x = x + self.mlp(self.norm2(x))
        return x

class PatchMerging(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.merge = nn.Conv2d(input_dim, output_dim, kernel_size=2, stride=2)
    
    def forward(self, x):
        return self.merge(x)

class SwinTransformer(nn.Module):
    def __init__(self, img_size, patch_size, num_classes, embed_dim, depth, num_heads, window_size):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.stage1 = nn.ModuleList([
            SwinBlock(embed_dim, num_heads, window_size, shift=(i % 2 == 1)) for i in range(depth)
        ])
        self.merge = PatchMerging(embed_dim, embed_dim * 2)
        self.stage2 = nn.ModuleList([
            SwinBlock(embed_dim * 2, num_heads, window_size, shift=(i % 2 == 1)) for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim * 2)
        self.head = nn.Linear(embed_dim * 2, num_classes)
    
    def forward(self, x):
        x = self.patch_embed(x)
        H, W = x.shape[2], x.shape[3]
        x = rearrange(x, "b c h w -> b (h w) c")
        for block in self.stage1:
            x = block(x, H, W)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        x = self.merge(x)
        H, W = x.shape[2], x.shape[3]
        x = rearrange(x, "b c h w -> b (h w) c")
        for block in self.stage2:
            x = block(x, H, W)
        x = self.norm(x.mean(dim=1))
        return self.head(x)

# Student Model
class SwinTransformerStudent(nn.Module):
    def __init__(self, img_size, patch_size, num_classes, embed_dim, depth, num_heads, window_size):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim // 2, kernel_size=patch_size, stride=patch_size)
        self.stage1 = nn.ModuleList([
            SwinBlock(embed_dim // 2, num_heads // 2, window_size, shift=(i % 2 == 1)) for i in range(depth // 2)
        ])
        self.merge = PatchMerging(embed_dim // 2, embed_dim)
        self.stage2 = nn.ModuleList([
            SwinBlock(embed_dim, num_heads // 2, window_size, shift=(i % 2 == 1)) for i in range(depth // 2)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        x = self.patch_embed(x)
        H, W = x.shape[2], x.shape[3]
        x = rearrange(x, "b c h w -> b (h w) c")
        for block in self.stage1:
            x = block(x, H, W)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        x = self.merge(x)
        H, W = x.shape[2], x.shape[3]
        x = rearrange(x, "b c h w -> b (h w) c")
        for block in self.stage2:
            x = block(x, H, W)
        x = self.norm(x.mean(dim=1))
        return self.head(x)

# Pruning function
def prune_model(model, pruning_amount=0.3):
    # Apply pruning
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            prune.l1_unstructured(module, name='weight', amount=pruning_amount)
            if hasattr(module, 'bias') and module.bias is not None:
                prune.l1_unstructured(module, name='bias', amount=pruning_amount)
    
    # Make pruning permanent
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            prune.remove(module, 'weight')
            if hasattr(module, 'bias') and module.bias is not None:
                prune.remove(module, 'bias')

    # Count non-zero parameters after pruning
    total_params = 0
    non_zero_params = 0
    for p in model.parameters():
        total_params += p.numel()
        non_zero_params += torch.count_nonzero(p).item()
    sparsity = 1 - non_zero_params / total_params
    print(f"Post-pruning sparsity: {sparsity:.2%} (Non-zero params: {non_zero_params}/{total_params})")
    return non_zero_params

# Distillation Loss
def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
    soft_targets = nn.functional.softmax(teacher_logits / temperature, dim=-1)
    soft_loss = nn.functional.kl_div(
        nn.functional.log_softmax(student_logits / temperature, dim=-1),
        soft_targets,
        reduction='batchmean'
    ) * (temperature ** 2)
    hard_loss = nn.functional.cross_entropy(student_logits, labels)
    return alpha * soft_loss + (1 - alpha) * hard_loss

def compute_metrics(model, dummy_input, device, system_name, stage=""):
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / (1024 * 1024)
    flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
    flops_per_inference_gflops = flops / 1e9
    with torch.no_grad():
        start_time = time.time()
        for _ in range(100):
            _ = model(dummy_input)
        inference_time = (time.time() - start_time) / 100
    
    print(f"\n=== Model Stats ({system_name} - {stage}) ===")
    print(f"Total Parameters: {total_params}")
    print(f"Model Size: {model_size_mb:.2f} MB")
    print(f"FLOPs Per Inference Step: {flops_per_inference_gflops:.2f} GFLOPs")
    print(f"Inference Time per Image: {inference_time:.4f} sec")
    return total_params, model_size_mb, flops, flops_per_inference_gflops, inference_time

def train_and_evaluate_with_pruning_distillation(teacher_model, student_model, trainloader, testloader, criterion, optimizer_teacher, optimizer_student, scheduler_teacher, scheduler_student, device, num_epochs, system_name):
    # Compute initial metrics for teacher
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    total_params_teacher, model_size_mb_teacher, flops_teacher, flops_per_inference_gflops_teacher, inference_time_teacher = compute_metrics(teacher_model, dummy_input, device, system_name, "Teacher")

    # Train teacher model
    print(f"\nTraining Teacher Model ({system_name})...")
    train_losses_teacher, test_losses_teacher, train_accuracies_teacher, test_accuracies_teacher = [], [], [], []
    epoch_times_teacher, peak_memory_usages_teacher = [], []
    total_train_time_teacher = 0

    for epoch in range(num_epochs):
        epoch_start = time.time()
        teacher_model.train()
        running_loss, correct, total = 0.0, 0, 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device=device)
        
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer_teacher.zero_grad()
            outputs = teacher_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_teacher.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_losses_teacher.append(running_loss / len(trainloader))
        train_accuracies_teacher.append(100 * correct / total)
        
        teacher_model.eval()
        test_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = teacher_model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_losses_teacher.append(test_loss / len(testloader))
        test_accuracies_teacher.append(100 * correct / total)

        scheduler_teacher.step()
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        total_train_time_teacher += epoch_time
        epoch_times_teacher.append(epoch_time)
        
        if device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated(device=device) / (1024 ** 3)
            peak_memory_usages_teacher.append(peak_memory)
            torch.cuda.reset_peak_memory_stats(device=device)
        else:
            peak_memory = 0
            peak_memory_usages_teacher.append(peak_memory)
        
        samples_per_epoch = len(trainloader.dataset)
        tflops_teacher = (flops_teacher * samples_per_epoch * 3) / (epoch_time * 1e12)
        
        print(f"\nEpoch {epoch+1}/{num_epochs} (Teacher - {system_name})")
        print(f"Train Loss: {train_losses_teacher[-1]:.4f}, Test Loss: {test_losses_teacher[-1]:.4f}")
        print(f"Train Acc: {train_accuracies_teacher[-1]:.2f}%, Test Acc: {test_accuracies_teacher[-1]:.2f}%")
        print(f"LR: {scheduler_teacher.get_last_lr()[0]:.6f}")
        print(f"Time Taken for Epoch: {epoch_time:.2f} sec")
        print(f"Peak Memory Usage: {peak_memory:.2f} GB")
        print(f"Estimated TFLOPS: {tflops_teacher:.2f}")

    # Apply pruning to teacher
    print(f"\nApplying pruning to Teacher Model ({system_name})...")
    non_zero_params_pruned = prune_model(teacher_model, pruning_amount=0.3)
    total_params_pruned, model_size_mb_pruned, flops_pruned, flops_per_inference_gflops_pruned, inference_time_pruned = compute_metrics(teacher_model, dummy_input, device, system_name, "Pruned Teacher")
    total_params_pruned = non_zero_params_pruned  # Update with non-zero params
    model_size_mb_pruned = total_params_pruned * 4 / (1024 * 1024)  # Recalculate size

    # Compute initial metrics for student
    total_params_student, model_size_mb_student, flops_student, flops_per_inference_gflops_student, inference_time_student = compute_metrics(student_model, dummy_input, device, system_name, "Student")

    # Train student with knowledge distillation
    print(f"\nTraining Student Model with Distillation ({system_name})...")
    train_losses_student, test_losses_student, train_accuracies_student, test_accuracies_student = [], [], [], []
    epoch_times_student, peak_memory_usages_student = [], []
    total_train_time_student = 0

    for epoch in range(num_epochs):
        epoch_start = time.time()
        student_model.train()
        teacher_model.eval()
        running_loss, correct, total = 0.0, 0, 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device=device)
        
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer_student.zero_grad()
            student_outputs = student_model(inputs)
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)
            loss = distillation_loss(student_outputs, teacher_outputs, labels)
            loss.backward()
            optimizer_student.step()
            running_loss += loss.item()
            _, predicted = torch.max(student_outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_losses_student.append(running_loss / len(trainloader))
        train_accuracies_student.append(100 * correct / total)
        
        student_model.eval()
        test_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = student_model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_losses_student.append(test_loss / len(testloader))
        test_accuracies_student.append(100 * correct / total)

        scheduler_student.step()
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        total_train_time_student += epoch_time
        epoch_times_student.append(epoch_time)
        
        if device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated(device=device) / (1024 ** 3)
            peak_memory_usages_student.append(peak_memory)
            torch.cuda.reset_peak_memory_stats(device=device)
        else:
            peak_memory = 0
            peak_memory_usages_student.append(peak_memory)
        
        samples_per_epoch = len(trainloader.dataset)
        tflops_student = (flops_student * samples_per_epoch * 3) / (epoch_time * 1e12)
        
        print(f"\nEpoch {epoch+1}/{num_epochs} (Student - {system_name})")
        print(f"Train Loss: {train_losses_student[-1]:.4f}, Test Loss: {test_losses_student[-1]:.4f}")
        print(f"Train Acc: {train_accuracies_student[-1]:.2f}%, Test Acc: {test_accuracies_student[-1]:.2f}%")
        print(f"LR: {scheduler_student.get_last_lr()[0]:.6f}")
        print(f"Time Taken for Epoch: {epoch_time:.2f} sec")
        print(f"Peak Memory Usage: {peak_memory:.2f} GB")
        print(f"Estimated TFLOPS: {tflops_student:.2f}")

    # Detailed Metrics Table
    print(f"\n=== Final Training Metrics Table ({system_name}) ===")
    print(f"{'Metric':<30} | {'Teacher':<15} | {'Pruned Teacher':<15} | {'Student':<15}")
    print("-" * 80)
    metrics = [
        ("Total Train Time (sec)", total_train_time_teacher, "-", total_train_time_student),
        ("Final Train Accuracy (%)", train_accuracies_teacher[-1], "-", train_accuracies_student[-1]),
        ("Final Test Accuracy (%)", test_accuracies_teacher[-1], "-", test_accuracies_student[-1]),
        ("Average Time per Epoch (sec)", sum(epoch_times_teacher)/len(epoch_times_teacher), "-", sum(epoch_times_student)/len(epoch_times_student)),
        ("Average Peak Memory (GB)", sum(peak_memory_usages_teacher)/len(peak_memory_usages_teacher), "-", sum(peak_memory_usages_student)/len(peak_memory_usages_student)),
        ("Average Estimated TFLOPS", sum((flops_teacher * len(trainloader.dataset) * 3) / (et * 1e12) for et in epoch_times_teacher)/len(epoch_times_teacher), "-", sum((flops_student * len(trainloader.dataset) * 3) / (et * 1e12) for et in epoch_times_student)/len(epoch_times_student)),
        ("Total Parameters", total_params_teacher, total_params_pruned, total_params_student),
        ("Model Size (MB)", model_size_mb_teacher, model_size_mb_pruned, model_size_mb_student),
        ("FLOPs Per Inference (GFLOPs)", flops_per_inference_gflops_teacher, flops_per_inference_gflops_pruned, flops_per_inference_gflops_student),
        ("Inference Time per Image (sec)", inference_time_teacher, inference_time_pruned, inference_time_student)
    ]
    for metric, teacher_val, pruned_val, student_val in metrics:
        if metric == "Inference Time per Image (sec)":
            teacher_str = f"{teacher_val:.4f}" if isinstance(teacher_val, float) else str(teacher_val)
            pruned_str = f"{pruned_val:.4f}" if isinstance(pruned_val, float) else str(pruned_val)
            student_str = f"{student_val:.4f}" if isinstance(student_val, float) else str(student_val)
        else:
            teacher_str = f"{teacher_val:.2f}" if isinstance(teacher_val, float) else str(teacher_val)
            pruned_str = f"{pruned_val:.2f}" if isinstance(pruned_val, float) else str(pruned_val)
            student_str = f"{student_val:.2f}" if isinstance(student_val, float) else str(student_val)
        print(f"{metric:<30} | {teacher_str:<15} | {pruned_str:<15} | {student_str:<15}")

    # Save plots
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses_teacher, label='Teacher Train Loss')
    plt.plot(range(1, num_epochs+1), test_losses_teacher, label='Teacher Test Loss')
    plt.plot(range(1, num_epochs+1), train_losses_student, label='Student Train Loss')
    plt.plot(range(1, num_epochs+1), test_losses_student, label='Student Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss Curves ({system_name})')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), train_accuracies_teacher, label='Teacher Train Accuracy')
    plt.plot(range(1, num_epochs+1), test_accuracies_teacher, label='Teacher Test Accuracy')
    plt.plot(range(1, num_epochs+1), train_accuracies_student, label='Student Train Accuracy')
    plt.plot(range(1, num_epochs+1), test_accuracies_student, label='Student Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Accuracy Curves ({system_name})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{system_name}_plots.png')
    plt.close()

    torch.save(teacher_model.state_dict(), f'{system_name}_teacher_pruned.pth')
    torch.save(student_model.state_dict(), f'{system_name}_student.pth')
    print(f"\nModels saved to '{system_name}_teacher_pruned.pth' and '{system_name}_student.pth'")

if __name__ == "__main__":
    model_config = {
        "Image_Size": "32x32",
        "Patch_Size": 4,
        "Embedding_Dim": 96,
        "Layers": 4,
        "Heads": 6,
        "Batch_Size": 128,
        "Learning_Rate": 0.001,
        "Epochs": 40,
        "Window_Size": 4,
        "Channels": 3,
        "Dataset": "CIFAR10"
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),  # Added rotation
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=model_config["Batch_Size"], shuffle=True, num_workers=4)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=model_config["Batch_Size"], shuffle=False, num_workers=4)
    
    teacher_model = SwinTransformer(
        img_size=int(model_config["Image_Size"].split("x")[0]),
        patch_size=model_config["Patch_Size"],
        num_classes=10,
        embed_dim=model_config["Embedding_Dim"],
        depth=model_config["Layers"],
        num_heads=model_config["Heads"],
        window_size=model_config["Window_Size"]
    ).to(device)
    
    student_model = SwinTransformerStudent(
        img_size=int(model_config["Image_Size"].split("x")[0]),
        patch_size=model_config["Patch_Size"],
        num_classes=10,
        embed_dim=model_config["Embedding_Dim"],
        depth=model_config["Layers"],
        num_heads=model_config["Heads"],
        window_size=model_config["Window_Size"]
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer_teacher = optim.Adam(teacher_model.parameters(), lr=model_config["Learning_Rate"], weight_decay=1e-4)  # Added weight decay
    optimizer_student = optim.Adam(student_model.parameters(), lr=model_config["Learning_Rate"], weight_decay=1e-4)  # Added weight decay
    scheduler_teacher = optim.lr_scheduler.StepLR(optimizer_teacher, step_size=10, gamma=0.1)
    scheduler_student = optim.lr_scheduler.StepLR(optimizer_student, step_size=10, gamma=0.1)
    
    train_and_evaluate_with_pruning_distillation(
        teacher_model, student_model, trainloader, testloader, criterion,
        optimizer_teacher, optimizer_student, scheduler_teacher, scheduler_student,
        device, model_config["Epochs"], "System_2_Prune_Distill"
    )
