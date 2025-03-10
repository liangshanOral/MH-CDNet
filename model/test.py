"""
通过验证集验证准确率
"""

from dataset import TrafficSignDataset, custom_collate_fn
from model_map import TrafficSignEncoder #在这里改变需要测试的model文件
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

# 假设 indices.csv 和 traffic_sign_data.csv 的路径
index_file_path = 'data/final_val_indices.csv'
# index_file_path = 'data/pne_10_indices.csv'
csv_file_path = 'data/final_traffic_sign_data.csv'
map_file_path = 'data/high_precision_map.csv'

# 初始化编码器
batch_size = 8
coordinates_dim = 2  # 假设坐标维度为
image_dim = 512     # 假设图像特征维度为2048
event_dim = 4
hidden_dim = 512     # 隐藏层维度
output_dim = 1       # 输出维度（概率）

# 创建数据集
dataset = TrafficSignDataset(index_file_path, csv_file_path, map_file_path, max_history=3)

# 创建 DataLoader
test_loader = DataLoader(dataset, batch_size = batch_size, shuffle=True, collate_fn=custom_collate_fn, drop_last=True)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

clip_model_path = "clip_model"
map_path = "clip_model/clip_finetuned.pth"

model = TrafficSignEncoder(coordinates_dim, image_dim, event_dim, clip_model_path, device)
model.to(device)


def test_model(model, test_loader, json_file):
    with open(json_file, 'r') as f:
        checkpoints = json.load(f)

    # 遍历每个 checkpoint 路径
    for checkpoint_info in checkpoints:
        model_path = checkpoint_info['model_path']
        
        # 判断文件是否存在
        if not os.path.exists(model_path):
            print(f"Checkpoint not found: {model_path}")
            continue

        checkpoint = torch.load(model_path)
        # 恢复模型和优化器状态
        model.load_state_dict(checkpoint['model_state_dict'])

        model.eval()  # 设置为评估模式
        predictions = []
        sig_predictions = []
        true_labels = []
        events = []
        trues = []

        with torch.no_grad():  # 禁用梯度计算
            for batch_idx, (now_info, history_info, map_info) in enumerate(test_loader):

                # 处理当前信息和历史信息，获取模型输出
                outputs = model(now_info, history_info, map_info)  # 输出形状为 (batch_size, 1)
                sig_outputs = torch.sigmoid(outputs).cpu().numpy()  # 这一步加上 sigmoid 转换
                outputs = outputs.cpu().numpy()  # 转为 NumPy 数组以便进一步处理

                # 提取 target (confidence)
                targets = [now_info[i]['confidence'] for i in range(batch_size)]

                for i in range(batch_size):
                    # 提取第 i 个样本的预测值和真实值
                    predicted_probability = outputs[i][0]
                    sig_predicted_probability = sig_outputs[i][0]
                    actual_probability = targets[i]

                    event = now_info[i]['event_type']

                    if sig_predicted_probability > 0.5 and (event == 'tp' or event == 'np'):
                        true = 1 # 检测成功
                    elif sig_predicted_probability < 0.5 and (event == 'fp' or event == 'tn'):
                        true = 1 # 检测成功
                    else:
                        true = 0 #检测错误率

                    # 保存预测、真实标签和事件类型
                    predictions.append(predicted_probability)
                    sig_predictions.append(sig_predicted_probability)
                    true_labels.append(actual_probability)
                    events.append(event)
                    trues.append(true)

        # 创建 DataFrame，将预测和真实标签配对
        results_df = pd.DataFrame({
            'Predicted Probability': predictions,
            'Sig Predicted Probability': sig_predictions,
            'Actual Probability': true_labels,
            'Event Type': events,
            'If True': trues,
        })

        # 按 'Event Type' 和 'If True'（true标签）进行分组
        accuracy_df = results_df.groupby(['Event Type', 'If True']).size().unstack(fill_value=0)

        # 计算每个事件类型的准确率
        accuracy_df['accuracy'] = accuracy_df[1] / (accuracy_df[0] + accuracy_df[1])

        # 显示准确率表
        print(accuracy_df)

        value_counts = results_df['If True'].value_counts()
        count_1 = value_counts.get(1, 0)  # 如果没有 1 则默认值为 0
        count_2 = value_counts.get(2, 0)  # 如果没有 2 则默认值为 0
        total_count = len(results_df)

        accuracy = ( count_1 + count_2 ) / total_count

        # 输出比例
        print(f"Ratio: {accuracy:.4f}, ckpt: {model_path}")
        
        # 保存到 CSV 文件
        # results_df.to_csv("data/test_results_GRU.csv", index=False)


def test_model_single(model_path, test_loader):
        
    # 判断文件是否存在
    if not os.path.exists(model_path):
        print(f"Checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path)
    # 恢复模型和优化器状态
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()  # 设置为评估模式
    predictions = []
    sig_predictions = []
    true_labels = []
    events = []
    trues = []

    with torch.no_grad():  # 禁用梯度计算
        for batch_idx, (now_info, history_info, map_info) in enumerate(test_loader):

            # 处理当前信息和历史信息，获取模型输出
            outputs = model(now_info, history_info, map_info)  # 输出形状为 (batch_size, 1)
            sig_outputs = torch.sigmoid(outputs).cpu().numpy()  # 这一步加上 sigmoid 转换
            outputs = outputs.cpu().numpy()  # 转为 NumPy 数组以便进一步处理

            # 提取 target (confidence)
            targets = [now_info[i]['confidence'] for i in range(batch_size)]

            for i in range(batch_size):
                # 提取第 i 个样本的预测值和真实值
                predicted_probability = outputs[i][0]
                sig_predicted_probability = sig_outputs[i][0]
                actual_probability = targets[i]

                event = now_info[i]['event_type']

                if sig_predicted_probability > 0.5 and (event == 'tp' or event == 'np'):
                    true = 1 # 检测成功
                elif sig_predicted_probability < 0.5 and (event == 'fp' or event == 'tn'):
                    true = 1 # 检测成功
                else:
                    true = 0 #检测错误率

                # 保存预测、真实标签和事件类型
                predictions.append(predicted_probability)
                sig_predictions.append(sig_predicted_probability)
                true_labels.append(actual_probability)
                events.append(event)
                trues.append(true)

    # 创建 DataFrame，将预测和真实标签配对
    results_df = pd.DataFrame({
        'Predicted Probability': predictions,
        'Sig Predicted Probability': sig_predictions,
        'Actual Probability': true_labels,
        'Event Type': events,
        'If True': trues,
    })

    # 按 'Event Type' 和 'If True'（true标签）进行分组
    accuracy_df = results_df.groupby(['Event Type', 'If True']).size().unstack(fill_value=0)

    # 计算每个事件类型的准确率
    accuracy_df['accuracy'] = accuracy_df[1] / (accuracy_df[0] + accuracy_df[1])

    # 显示准确率表
    print(accuracy_df)

    value_counts = results_df['If True'].value_counts()
    count_1 = value_counts.get(1, 0)  # 如果没有 1 则默认值为 0
    count_2 = value_counts.get(2, 0)  # 如果没有 2 则默认值为 0
    total_count = len(results_df)

    accuracy = ( count_1 + count_2 ) / total_count

    # 输出比例
    print(f"Ratio: {accuracy:.4f}, ckpt: {model_path}")

    # 保存到 CSV 文件
    results_df.to_csv('data/test.csv', index=False)



if __name__ == "__main__":
    
    #train_model(model, train_loader, device)
    # json_file = "training_log_0110.json"
    # test_model(model, test_loader, json_file)
    
    model_path = 'ckpt/model_checkpoint_lr1e-05_wd1e-05_epoch60_0111.pth'
    test_model_single(model_path,test_loader)