import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pandas as pd
import csv

from models import *
from models.vit import ViT, channel_selection
from models.vit_slim import ViT_slim
from utils import progress_bar
from warmup_scheduler import GradualWarmupScheduler
import albumentations
import psutil
import time
import os
from thop import profile
def calculate_accuracy(model, dataloader):
    correct = total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            # print(images.shape)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# --- 2. Inference Speed and Latency ---
def benchmark_inference(model, input_shape=(1, 3, 32, 32), runs=100):
    dummy_input = torch.randn(input_shape).to(device)
    torch.cuda.synchronize()
    start = time.time()
    for i in range(runs):
        output = model(dummy_input)
    torch.cuda.synchronize()
    total_time = time.time() - start
    latency = total_time / runs
    throughput = runs / total_time
    return latency, throughput

# --- 3. Model Size ---
def get_model_size(model, temp_path='temp.pth'):
    torch.save(model.state_dict(), temp_path)
    size_mb = os.path.getsize(temp_path) / 1e6
    os.remove(temp_path)
    return size_mb

# --- 4. Memory Usage (estimated by RAM during execution) ---
def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1e6  # in MB
    return mem

# --- 5. FLOPs and Parameters ---
def get_flops(model, input_shape=(1, 3, 32, 32)):
    dummy_input = torch.randn(input_shape).to(device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    return flops / 1e9, params / 1e6  # GFLOPs and MParams

# --- 6. Estimate Power Usage ---
def estimate_power(flops, latency):
    # Rough estimate: 1 GFLOP = ~0.1 Watt-sec (example heuristic)
    energy = flops * 0.1  # Watt-seconds
    power = energy / latency  # Watts
    return power

def compute_metrics(model,testloader):
    # --- Run all metrics ---
    accuracy = calculate_accuracy(model, testloader)
    latency, speed = benchmark_inference(model)
    model_size = get_model_size(model)
    mem_usage = get_memory_usage()
    flops, params = get_flops(model)
    power = estimate_power(flops, latency)


    # --- Print results ---
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Inference Latency: {latency*1000:.2f} ms")
    print(f"Inference Speed: {speed:.2f} samples/sec")
    print(f"Model Size: {model_size:.2f} MB")
    print(f"Memory Usage (runtime): {mem_usage:.2f} MB")
    print(f"FLOPs: {flops:.2f} GFLOPs")
    print(f"Parameters: {params:.2f} Million")
    print(f"Estimated Power: {power:.2f} Watts")

# Define distillation loss
class DistillLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.7):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, targets):
        distill = self.kl_div(
            F.log_softmax(student_logits / self.T, dim=1),
            F.softmax(teacher_logits / self.T, dim=1),
        ) * (self.T ** 2)

        ce = self.ce_loss(student_logits, targets)
        return self.alpha * distill + (1 - self.alpha) * ce


def train(epoch,teacher,student,optimizer,criterion):
    print('\nEpoch: %d' % epoch)
    student.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            teacher_outputs = teacher(inputs)
        student_outputs = student(inputs)
        loss = criterion(student_outputs, teacher_outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = student_outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1)

def test(epoch, student,scheduler,optimizer):
    global best_acc
    student.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion_ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = student(inputs)
            loss = criterion_ce(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Update scheduler
    scheduler.step(test_loss)
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': student.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+'vit-4-'+'ckpt_student_kt.t7')
        best_acc = acc
    
    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    return test_loss, acc


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cudnn.benchmark = True

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=8)
best_acc = 0
teacher = ViT(
    image_size=32, 
    patch_size=4, 
    num_classes=10, 
    dim=512,
    depth=6, 
    heads=8, 
    mlp_dim=512, 
    dropout=0.1, 
    emb_dropout=0.1
)

teacher.load_state_dict(torch.load("checkpoint/vit-4-ckpt_512.t7")["net"])

student = ViT(
    image_size=32, 
    patch_size=4, 
    num_classes=10, 
    dim=256,
    depth=4, 
    heads=4, 
    mlp_dim=256, 
    dropout=0.1, 
    emb_dropout=0.1
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
teacher = teacher.to(device)
student = student.to(device)
teacher.eval()
criterion = DistillLoss()
optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
from torch.optim import lr_scheduler
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True, min_lr=1e-3*1e-5, factor=0.1)


list_loss = []
list_acc = []
for epoch in range(0,100):
    trainloss = train(epoch,teacher,student,optimizer,criterion)
    val_loss, acc = test(epoch,student,scheduler,optimizer)
    
    list_loss.append(val_loss)
    list_acc.append(acc)

compute_metrics(student,testloader)