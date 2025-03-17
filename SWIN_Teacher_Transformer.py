import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from einops import rearrange
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import time
from thop import profile  # For FLOPs calculation

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
        
        # Relative position bias
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
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(0.1)
    
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
        # Stage 1
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.stage1 = nn.ModuleList([
            SwinBlock(embed_dim, num_heads, window_size, shift=(i % 2 == 1)) for i in range(depth)
        ])
        # Patch Merging to Stage 2
        self.merge = PatchMerging(embed_dim, embed_dim * 2)
        # Stage 2
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
        "Dataset": "CIFAR-10"
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=model_config["Batch_Size"], shuffle=True, num_workers=4)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=model_config["Batch_Size"], shuffle=False, num_workers=4)
    
    model = SwinTransformer(
        img_size=int(model_config["Image_Size"].split("x")[0]),
        patch_size=model_config["Patch_Size"],
        num_classes=10,
        embed_dim=model_config["Embedding_Dim"],
        depth=model_config["Layers"],
        num_heads=model_config["Heads"],
        window_size=model_config["Window_Size"]
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=model_config["Learning_Rate"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Compute model stats
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / (1024 * 1024)
    max_len = model_config["Window_Size"] * model_config["Window_Size"]  # Tokens per window
    
    # Compute FLOPs and inference time
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
    flops_per_inference_gflops = flops / 1e9  # Convert to GFLOPs
    
    # Measure inference time per image
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        for _ in range(100):  # Average over 100 runs
            _ = model(dummy_input)
        inference_time = (time.time() - start_time) / 100
    
    print("\n=== Model Configuration ===")
    for key, value in model_config.items():
        print(f"{key}: {value}")
    print(f"Channels: {model_config['Channels']}")
    print(f"max_len (Tokens per Window): {max_len}")
    
    print("\n=== Model Stats ===")
    print(f"Total Parameters: {total_params}")
    print(f"Model Size: {model_size_mb:.2f} MB")
    print(f"FLOPs Per Inference Step: {flops_per_inference_gflops:.2f} GFLOPs")
    print(f"Inference Time per Image: {inference_time:.4f} sec")
    
    num_epochs = model_config["Epochs"]
    train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []
    epoch_times, peak_memory_usages = [], []
    total_train_time = 0

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_losses.append(running_loss / len(trainloader))
        train_accuracies.append(100 * correct / total)
        
        model.eval()
        test_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_losses.append(test_loss / len(testloader))
        test_accuracies.append(100 * correct / total)

        scheduler.step()
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        total_train_time += epoch_time
        epoch_times.append(epoch_time)
        
        # Memory usage
        if device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert to GB
            peak_memory_usages.append(peak_memory)
            torch.cuda.reset_peak_memory_stats()
        else:
            peak_memory = 0
            peak_memory_usages.append(peak_memory)
        
        # Estimate TFLOPS (per epoch)
        samples_per_epoch = len(trainset)
        tflops = (flops * samples_per_epoch * 3) / (epoch_time * 1e12)  # 3 for forward+backward
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}")
        print(f"Train Acc: {train_accuracies[-1]:.2f}%, Test Acc: {test_accuracies[-1]:.2f}%")
        print(f"LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"Time Taken for Epoch: {epoch_time:.2f} sec")
        print(f"Peak Memory Usage: {peak_memory:.2f} GB")
        print(f"Estimated TFLOPS: {tflops:.2f}")

    # Final metrics
    print("\n=== Final Training Metrics ===")
    print(f"Total Train Time: {total_train_time:.2f} sec")
    print(f"Final Train Accuracy: {train_accuracies[-1]:.2f}%")
    print(f"Final Test Accuracy: {test_accuracies[-1]:.2f}%")
    print(f"Average Time per Epoch: {sum(epoch_times)/len(epoch_times):.2f} sec")
    print(f"Average Peak Memory Usage per Epoch: {sum(peak_memory_usages)/len(peak_memory_usages):.2f} GB")
    print(f"Average Estimated TFLOPS: {sum((flops * len(trainset) * 3) / (et * 1e12) for et in epoch_times)/len(epoch_times):.2f}")

    # Plotting
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, num_epochs+1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curves')
    plt.legend()
    plt.tight_layout()
    plt.show()

    torch.save(model.state_dict(), 'swin_level2_cifar10.pth')
    print("Model saved to 'swin_level2_cifar10.pth'")
