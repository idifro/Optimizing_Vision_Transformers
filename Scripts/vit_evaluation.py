import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms
import threading
import time
import os
import argparse
import pandas as pd
import csv

from models import *
from models.vit import ViT, channel_selection
from models.vit_slim import ViT_slim
from utils import progress_bar
from thop import profile
from nvitop import Device

import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cudnn.benchmark = True


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)


class PowerMonitor(threading.Thread):
    def __init__(self, interval=0.1, device_index=0):
        super().__init__()
        self.interval = interval
        self.device = Device(index=device_index)
        self.running = False
        self.power_samples = []
        self.timestamps = []

    def run(self):
        self.running = True
        while self.running:
            try:
                power = self.device.power_draw()  # in milliwatts
                self.power_samples.append(power)
                self.timestamps.append(time.time())
                time.sleep(self.interval)
            except Exception as e:
                print(f"[PowerMonitor] Error: {e}")
                break

    def stop(self):
        self.running = False

    def get_results(self):
        return self.timestamps, self.power_samples
    
def estimate_power(model):
    input_shape = (100,3,32,32)
    dummy_input = torch.randn(input_shape).to(device)
    torch.cuda.empty_cache()
    # Start power monitor in background
    monitor = PowerMonitor(interval=0.1)
    monitor.start()
    # epoch 100, 1 batch_size (100) 
    for _ in range(100):
        _ = model(dummy_input)
    # Stop monitoring
    monitor.stop()
    monitor.join()
    # Retrieve power data
    timestamps, power_samples = monitor.get_results()
    return timestamps, power_samples

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

def log_resource_usage(tag=""):
    import os, psutil, torch
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / (1024 ** 2)
    gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)
    gpu_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"[{tag}] CPU RAM: {cpu_mem:.10f} MB | GPU RAM: {gpu_mem:.10f} MB (Peak: {gpu_peak:.10f} MB)")
    return cpu_mem,gpu_mem,gpu_peak

def compute_inference_resource_usage(tag,model_path,save_method,input_shape=(100, 3, 32, 32)):
    
    torch.cuda.empty_cache()
    dummy_input = torch.randn(input_shape).to(device)
    # # Warm-up
    # for _ in range(50):
    #     _ = model(dummy_input)
    torch.cuda.empty_cache()
    # Load model set it to eval and to GPU
    if save_method == "state_dict":
        model = ViT(
            image_size = 32,
            patch_size = 4,
            num_classes = 10,
            dim = 512,                  # 512
            depth = 6,
            heads = 8,
            mlp_dim = 512,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['acc']
        model.load_state_dict(checkpoint['net'])
        model.to(device)
        model.eval()
        print("=> loading checkpoint '{}'".format(model_path))
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}".format(model_path, checkpoint['epoch'], best_prec1))
    elif save_method == "whole":
        model = torch.load(model_path,weights_only=False)
        model.to(device)
        model.eval()
        print("=> loading checkpoint '{}'".format(model_path))
    
    torch.cuda.empty_cache()
    # Before Inference
    cpu_mem_b,gpu_mem_b,gpu_peak_b = log_resource_usage(f"{tag} - before inference")
    _ = model(dummy_input)
    # After inference 
    cpu_mem_a,gpu_mem_a,gpu_peak_a = log_resource_usage(f"{tag} - after inference")
    return model,(cpu_mem_a-cpu_mem_b),(gpu_mem_a-gpu_mem_b),(gpu_peak_a-gpu_peak_b)


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

def benchmark_inference(model,input_shape=(1, 3, 32, 32), runs=100):
    dummy_input = torch.randn(input_shape).to(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Warm-up
    for _ in range(50):
        _ = model(dummy_input)

    # Timed run
    # GPU level timing for more accuracy
    latencies = []
    for _ in range(runs):
        start.record()
        _ = model(dummy_input)
        end.record()
        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end))

    avg_latency = sum(latencies) / len(latencies)
    latency_sec = avg_latency / 1000
    avg_throughput = len(latencies)/ latency_sec
    return avg_latency, avg_throughput


def get_model_size(model, temp_path='temp.pth'):
    torch.save(model.state_dict(), temp_path)
    size_mb = os.path.getsize(temp_path) / 1e6
    os.remove(temp_path)
    return size_mb

# --- 5. FLOPs and Parameters ---
def get_flops(model, input_shape=(1, 3, 32, 32)):
    dummy_input = torch.randn(input_shape).to(device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    return flops / 1e9, params / 1e6  # GFLOPs and MParams

# def estimate_power(model):
#     input_shape = (100,3,32,32)
#     dummy_input = torch.randn(input_shape).to(device)
#     torch.cuda.empty_cache()
#     # Warm-up
#     for _ in range(50):
#         _ = model(dummy_input)
    
#     dev_1 = Device(index=0)
#     power_usage_before = dev_1.power_draw() #in milliwatts
#     # epoch 100, 1 batch_size (100) 
#     for _ in range(100):
#         _ = model(dummy_input)
#     power_usage_after = dev_1.power_draw()
#     return (power_usage_after - power_usage_before)/1000 # in watts

# def estimate_mean_power(model):
#     power_usage_list = []
#     # for i in range(10):
#     #     power_used = estimate_power(model)
#     #     power_usage_list.append(power_used)
#     power_used = estimate_power(model)
#     # return np.mean(power_usage_list)
#     return power_used
        
def compute_evaluation_metrics(model_path,model_name):
    model,cpu_mem,gpu_mem,gpu_peak = compute_inference_resource_usage(model_name,model_path,save_method="whole")
    accuracy = calculate_accuracy(model,testloader)
    latency,speed = benchmark_inference(model)
    flops, params = get_flops(model)
    timestamps, power_samples = estimate_power(model)
    avg_power_w = np.mean(power_samples) / 1000
    print(f"{model_name} accuracy is : {accuracy}")
    print(f"{model_name} 1 batch metric CPU Mem Diff: {cpu_mem:.2f} MB, GPU Mem Diff: {gpu_mem:.2f} MB")
    print(f"{model_name} Inference Latency: {latency:.2f} ms")
    print(f"{model_name} Inference Speed: {speed:.2f} samples/sec")
    print(f"{model_name} FLOPs: {flops:.2f} GFLOPs")
    print(f"{model_name} Parameters: {params:.2f} Million")
    print(f"Average GPU Power Usage: {avg_power_w:.2f} W")
    # print(f"{model_name} Power used per batch inference 100 epochs :{power_used:.2f} watts")
    return timestamps, power_samples



# model_path = "checkpoint/vit-4-ckpt_512.t7"
# model,cpu_mem,gpu_mem,gpu_peak = compute_inference_resource_usage("baseline_vit",model_path,save_method="state_dict")

model_path = "checkpoint/model_baseline_vit_512.pth"
timestamps, power_samples = compute_evaluation_metrics(model_path,"baseline_vit")

plt.plot(timestamps, np.array(power_samples)/1000)
plt.xlabel("Time (s)")
plt.ylabel("Power (W)")
plt.title("GPU Power Usage Over Time")
plt.grid(True)
plt.savefig("baseline_time_vs_power.png")
# model_path = "checkpoint/model_prune_20.pth"
# compute_evaluation_metrics(model_path,"Pruned_20")
# model_path = "checkpoint/model_prune_30.pth"
# compute_evaluation_metrics(model_path,"Pruned_30")
# model_path = "checkpoint/model_prune_40.pth"
# compute_evaluation_metrics(model_path,"Pruned_40")

# model_path = "checkpoint/model_pruned_40_student.pth"
# compute_evaluation_metrics(model_path,"Pruned_40_KD")

# model_path = "checkpoint/model_student_kd.pth"
# compute_evaluation_metrics(model_path,"Student_KD")

