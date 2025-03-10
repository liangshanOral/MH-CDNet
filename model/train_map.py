from dataset import TrafficSignDataset, custom_collate_fn
from model_map import TrafficSignEncoder
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import CyclicLR
import random
import os
import json
from itertools import product
import warnings
warnings.filterwarnings("ignore")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果有多个 GPU，确保所有 GPU 都设置相同的种子
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 调用函数设置随机种子
set_seed(42)

class WeightedBCELoss(nn.Module):
    def __init__(self, positive_weight=1.0, negative_weight=1.0):
        super(WeightedBCELoss, self).__init__()
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight

    def forward(self, outputs, labels):

        # 计算每个样本的权重
        weights = torch.where(labels == 1, self.positive_weight, self.negative_weight)

        # 使用 BCEWithLogitsLoss（它内置 sigmoid 函数）
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')  # 不做 reduction，便于加权
        unweighted_loss = bce_loss(outputs, labels)

        # 加权损失
        weighted_loss = weights * unweighted_loss

        return weighted_loss.mean()  # 返回平均损失

class WeightedMSELoss(nn.Module):
    def __init__(self, positive_weight=1.0, negative_weight=7.0):
        super(WeightedMSELoss, self).__init__()
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight

    def forward(self, outputs, labels):
        # 将标签转换为二进制形式，判断正负样本
        binary_labels = (labels > 0.5).float()  # 标签为正样本则为1，否则为0
        
        # 计算每个样本的权重
        weights = torch.where(binary_labels == 1, self.positive_weight, self.negative_weight)
        
        # 计算加权 MSE 损失
        loss = weights * (outputs - labels) ** 2
        
        return loss.mean()  # 返回平均损失

class EarlyStopping:
    def __init__(self, patience=30, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None

    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# 在训练循环中使用早停法
early_stopping = EarlyStopping(patience=30, verbose=True)

# 初始化编码器
batch_size = 64 # 之前是512 2025.1.8
coordinates_dim = 2  # 坐标维度
event_dim = 4
image_dim = 512     # 图像特征维度为512(clip) 2048(resnet)
output_dim = 1       # 输出维度（概率）

# 假设 indices.csv 和 traffic_sign_data.csv 的路径
index_file_path = 'data/final_train_indices.csv'
test_index_file_path = 'data/final_test_indices.csv'
csv_file_path = 'data/final_traffic_sign_data.csv'
map_file_path = 'data/high_precision_map_2.csv'

# 创建数据集
dataset = TrafficSignDataset(index_file_path, csv_file_path, map_file_path, max_history=3)
test_dataset = TrafficSignDataset(test_index_file_path, csv_file_path, map_file_path, max_history=3)

# 创建 DataLoader
dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=True, collate_fn=custom_collate_fn, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=True, collate_fn=custom_collate_fn, drop_last=True)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# torch.cuda.set_device(3)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clip_model_path = "clip_model"
map_path = "clip_model/clip_finetuned.pth"

model = TrafficSignEncoder(coordinates_dim, image_dim, event_dim, clip_model_path, device)
model.to(device)

num_epochs = 60  # 设定训练的轮数
writer = SummaryWriter(log_dir='log/')  # 你可以指定想要的目录

# 创建超参数范围
learning_rates = [1e-5]
# learning_rates = [1e-8, 1e-3, 1e-2, 1e-1]
# weight_decays = [1e-6, 1e-4]
# learning_rates = [1e-5]
weight_decays = [1e-5]

# 定义日志文件路径
log_file = 'training_log_0111.json'

# 创建日志文件
if not os.path.exists(log_file):
    with open(log_file, 'w') as file:
        json.dump([], file)

for lr, wd in product(learning_rates, weight_decays):
    # 设置损失函数和优化器
    Loss = WeightedBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    for epoch in range(num_epochs):
        
        model.train()  # 设置为训练模式
        epoch_loss = 0  # 记录每个epoch的总损失

        for batch_idx, (now_info, history_info, map_info) in enumerate(dataloader):

            optimizer.zero_grad()  # 清除之前的梯度

            # 处理当前信息和历史信息
            outputs = model(now_info, history_info, map_info)
            outputs = outputs.squeeze(-1) 

            # 提取整个批次的 target (confidence)
            targets = [now_info[i]['existence'] for i in range(batch_size)]
            targets = torch.tensor(targets, dtype=torch.float32).to(device)
    
            loss = Loss(outputs, targets)

            loss.backward()  # 反向传播
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()  # 更新参数

            epoch_loss += loss.item()  # 累加损失

        # 计算平均损失
        avg_loss = epoch_loss / len(dataloader)

        # 学习率调度器更新
        scheduler.step()

        writer.add_scalar('Loss/train', avg_loss, epoch)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, (now_info, history_info, map_info) in enumerate(test_dataloader):

                # 处理当前信息和历史信息
                outputs = model(now_info, history_info, map_info)
                outputs = outputs.squeeze(-1) 

                # 提取整个批次的 target (confidence)
                targets = [now_info[i]['existence'] for i in range(batch_size)]
                targets = torch.tensor(targets, dtype=torch.float32).to(device)
        
                loss = Loss(outputs, targets)
                val_loss += loss.item()  # 累加损失
        
        val_loss /= len(test_dataloader)
        writer.add_scalar('Loss/test', val_loss, epoch)
        print(f"Epoch {epoch + 1}, Validation loss: {val_loss:.4f}")

        # 检查是否提前停止
        if early_stopping(avg_loss):
            print("Early stopping triggered")
            break

        # 每10个epoch保存模型
        # === 保存模型 ===
        if (epoch + 1) %10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'learning_rate': lr,
                'weight_decay': wd
            }
            model_path = f'ckpt/model_checkpoint_lr{lr}_wd{wd}_epoch{epoch + 1}_0111.pth'
            torch.save(checkpoint, model_path)
            print(f"Model checkpoint saved at {model_path}")

            # 记录日志
            log_entry = {
                "epoch": epoch + 1,
                "learning_rate": lr,
                "weight_decay": wd,
                "model_path": model_path
            }
            with open(log_file, 'r+') as file:
                logs = json.load(file)
                logs.append(log_entry)
                file.seek(0)
                json.dump(logs, file, indent=4)

writer.close()