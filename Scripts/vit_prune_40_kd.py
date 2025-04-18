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
import time

def test(model):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))


def get_prune_model(model,prune_percent):
    total = 0
    for m in model.modules():
        if isinstance(m, channel_selection):
            total += m.indexes.data.shape[0]
    
    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, channel_selection):
            size = m.indexes.data.shape[0]
            bn[index:(index+size)] = m.indexes.data.abs().clone()
            index += size
    
    y, i = torch.sort(bn)
    thre_index = int(total * prune_percent)
    thre = y[thre_index]

    pruned = 0
    cfg = []
    cfg_mask = []
    for k, m in enumerate(model.modules()):
        if isinstance(m, channel_selection):
            # print(k)
            # print(m)
            if k in [16,40,64,88,112,136]:
                weight_copy = m.indexes.data.abs().clone()
                mask = weight_copy.gt(thre).float().cuda()
                thre_ = thre.clone()
                while (torch.sum(mask)%8 !=0):                       # heads
                    thre_ = thre_ - 0.0001
                    mask = weight_copy.gt(thre_).float().cuda()
            else:
                weight_copy = m.indexes.data.abs().clone()
                mask = weight_copy.gt(thre).float().cuda()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.indexes.data.mul_(mask)
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                format(k, mask.shape[0], int(torch.sum(mask))))
    pruned_ratio = pruned/total
    print('Pre-processing Successful!')
    print("Pruned Ratio:",pruned_ratio)
    print(cfg)

    test(model)
    cfg_prune = []
    for i in range(len(cfg)):
        if i%2!=0:
            cfg_prune.append([cfg[i-1],cfg[i]])
    print(cfg_prune)

    newmodel = ViT_slim(image_size = 32,
    patch_size = 4,
    num_classes = 10,
    dim = 512,
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1,
    cfg=cfg_prune)
    newmodel.to(device)
    # num_parameters = sum([param.nelement() for param in newmodel.parameters()])

    newmodel_dict = newmodel.state_dict().copy()

    i = 0
    newdict = {}
    for k,v in model.state_dict().items():
        if 'net1.0.weight' in k:
            # print(k)
            # print(v.size())
            # print('----------')
            idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
            newdict[k] = v[idx.tolist()].clone()
        elif 'net1.0.bias' in k:
            # print(k)
            # print(v.size())
            # print('----------')
            idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
            newdict[k] = v[idx.tolist()].clone()
        elif 'to_q' in k or 'to_k' in k or 'to_v' in k:
            # print(k)
            # print(v.size())
            # print('----------')
            idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
            newdict[k] = v[idx.tolist()].clone()
        elif 'net2.0.weight' in k:
            # print(k)
            # print(v.size())
            # print('----------')
            idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
            newdict[k] = v[:,idx.tolist()].clone()
            i = i + 1
        elif 'to_out.0.weight' in k:
            # print(k)
            # print(v.size())
            # print('----------')
            idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
            newdict[k] = v[:,idx.tolist()].clone()
            i = i + 1

        elif k in newmodel.state_dict():
            newdict[k] = v

    newmodel_dict.update(newdict)
    newmodel.load_state_dict(newmodel_dict)

    return newmodel

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
model = model.to(device)

model_path = "checkpoint/vit-4-ckpt_512.t7"
print("=> loading checkpoint '{}'".format(model_path))
checkpoint = torch.load(model_path)
start_epoch = checkpoint['epoch']
best_prec1 = checkpoint['acc']
model.load_state_dict(checkpoint['net'])
print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}".format(model_path, checkpoint['epoch'], best_prec1))

prune_40 = get_prune_model(model,0.4)

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
teacher.eval()

# Perform knowledge distillation on prune_40
# Train with distillation
student = prune_40
device = 'cuda' if torch.cuda.is_available() else 'cpu'
teacher = teacher.to(device)
student = student.to(device)

criterion = DistillLoss()
optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
from torch.optim import lr_scheduler
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True, min_lr=1e-3*1e-5, factor=0.1)

def train(epoch):
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

best_acc = 0

def test(epoch):
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
        torch.save(state, './checkpoint/'+'model_'+'ckpt_pruned_student.t7')
        student.train()
        torch.save(student,"checkpoint/model_pruned_40_student.pth")
        student.eval()
        best_acc = acc
    
    # os.makedirs("log", exist_ok=True)
    # content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    # print(content)
    return test_loss, acc

list_loss = []
list_acc = []
for epoch in range(0,50):
    trainloss = train(epoch)
    val_loss, acc = test(epoch)
    
    list_loss.append(val_loss)
    list_acc.append(acc)
print("Model Training Finished")