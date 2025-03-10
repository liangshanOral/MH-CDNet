import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import imagehash
import cv2
import random
from datetime import datetime, timedelta
import torchvision.transforms as transforms
from torchvision import models
import pickle
import math


class FeatureExtractor:
    def __init__(self, clip_model_path, device='cuda:1'):
        self.device = device
        self.clip_model = CLIPModel.from_pretrained(clip_model_path).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(clip_model_path)

    def extract_img_features(self, image_path, bbox):
        """
        提取图像特征并根据边界框裁剪。

        :param image_path: 图像的路径
        :param bbox: 要裁剪的边界框 (left, upper, right, lower)
        :return: 图像特征张量
        """
        # 打开图片并进行裁剪
        img = Image.open(image_path).convert('RGB')
        cropped_img = img.crop(bbox)
        # 将 PIL 图像转换为张量
        transform = transforms.ToTensor()  # 创建一个转换对象
        cropped_img = transform(cropped_img).to(self.device)  # 添加 batch 维度并移动到设备

        # 使用 CLIP 的 processor 进行预处理
        inputs_images = self.processor(images=cropped_img, return_tensors="pt", do_rescale=False).to(self.device)

        # 提取图像特征
        with torch.no_grad():
            img_features = self.clip_model.get_image_features(**inputs_images)

        return img_features  # 返回 CLIP 模型的图像特征

feature_extractor = FeatureExtractor(clip_model_path = "clip_model")

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5):
        super(PositionalEncoding, self).__init__()
        # 创建一个位置编码矩阵，大小为 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 添加 batch_size 维度，形状变为 (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 获取输入的长度
        x_len = x.size(1)
        # 截取前 x_len 个位置编码并加到输入中
        return x + self.pe[:, :x_len, :]


class TrafficSignEncoder(nn.Module):
    def __init__(self, coordinates_dim, image_dim, event_dim, clip_model_path, device):
        super(TrafficSignEncoder, self).__init__()
        self.coordinates_dim = coordinates_dim 
        self.image_dim = image_dim
        self.event_dim = event_dim
        self.device = device

        self.clip_model = CLIPModel.from_pretrained(clip_model_path).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(clip_model_path)

        # 嵌入层
        self.his_combined_dim = 256
        self.map_combined_dim = 192
        self.event_linear = nn.Linear(self.event_dim, 64)
        self.coord_linear = nn.Linear(self.coordinates_dim, 64)
        self.image_linear = nn.Linear(self.image_dim, 128)

        self.bn_event = nn.BatchNorm1d(64)
        self.bn_coord = nn.BatchNorm1d(64)
        self.bn_image = nn.BatchNorm1d(128)

        # 创建 Transformer Encoder 模块
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.his_combined_dim, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.positional_encoding_layer = PositionalEncoding(d_model=self.his_combined_dim)
        
        # 输出层
        # 历史信息的全连接层
        self.history_fc1 = nn.Linear(self.his_combined_dim, 64)
        self.history_fc2 = nn.Linear(64, 32)
        self.history_fc3 = nn.Linear(32, 1)  # 最终输出为 (B, 1)

        # 地图信息的全连接层
        self.map_fc1 = nn.Linear(self.map_combined_dim, 64)
        self.map_fc2 = nn.Linear(64, 32)
        self.map_fc3 = nn.Linear(32, 1)  # 最终输出为 (B, 1)

        # 用于融合两个输出的 learnable 权重
        # 4指的是历史事件长度
        self.alpha_linear = nn.Linear(2, 1)

        # 调用初始化函数来初始化权重
        self._initialize_weights()
        # 30%的神经元会在每次前向传播中被随机丢弃
        self.dropout = nn.Dropout(p=0.3)  

    def _initialize_weights(self):
        # 遍历每个模块，找到所有线性层并应用合适的初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  # Kaiming 初始化
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)  # 偏置初始化为 0

    def forward(self, now_info, history_info, map_info):

        # 处理当前事件
        now_coordinates = torch.tensor([info['coordinates'][:2] for info in now_info], dtype=torch.float32).to(self.device)  # (B, 2)
        now_image_features = torch.stack([self.get_img_feature(info['image_path'], info['bbox']) for info in now_info]).squeeze(1)
        # 当前事件为阴性事件标志
        mask = (now_coordinates == torch.tensor([0.0, 0.0], device=now_coordinates.device)).all(dim=1)
        fixed_value = torch.tensor([199999.0, 199999.0], device=now_coordinates.device)
        
        # 处理历史信息
        # 图片
        history_image_features = torch.stack([
            torch.stack([
                self.get_img_feature(hist['image_path'], hist['bbox']) if hist['image_path'] is not None 
                else torch.zeros(1, now_image_features.shape[1]).to(self.device)  # 使用第二维的大小
                for hist in history
            ])
            for history in history_info
        ]).squeeze(2)
        batch_size, history_cnt, _ = history_image_features.shape

        now_image_features_expanded = now_image_features.unsqueeze(1).expand(-1, history_cnt, -1)
        
        his_image_diff = now_image_features_expanded - history_image_features
        his_image_diff_reshaped = his_image_diff.view(-1, self.image_dim)  # (B * history_cnt, event_dim)
        his_image_embedding = self.bn_image(self.image_linear(his_image_diff_reshaped * 1000))
        his_image_embedding = his_image_embedding.view(batch_size, history_cnt, -1)

        # 坐标
        history_coordinates = torch.tensor([[hist['coordinates'][:2] if hist['coordinates'] is not None else [0.0, 0.0] for hist in history] for history in history_info], dtype=torch.float32).to(self.device)
        # 将 now_coordinates 维度扩展为 (B, 1, 2)，然后在第二维度上扩展为 (B, history_cnts, 2)
        now_coordinates_expanded = now_coordinates.unsqueeze(1).expand(-1, history_cnt, -1)
        # 计算历史坐标之间的差异，now_coordinates_expanded 的形状为 (B, history_cnts, 2)
        his_coordinates_diff = torch.where(mask.unsqueeze(1).unsqueeze(2), fixed_value, now_coordinates_expanded - history_coordinates)
        his_coordinates_diff_reshaped = his_coordinates_diff.view(-1, self.coordinates_dim)  
        his_coord_embedding = self.bn_coord(self.coord_linear(his_coordinates_diff_reshaped))
        his_coord_embedding = his_coord_embedding.view(batch_size, history_cnt, -1)
     
        # 事件
        history_event_types = [
            [hist['event_type'] if hist['event_type'] is not None else None for hist in history]
            for history in history_info
        ]
        # 将事件类型转换为独热编码并构造历史事件类型张量
        event_type_mapping = {'tp': [1, 0, 0, 0], 'fp': [0, 1, 0, 0], 'tn': [0, 0, 1, 0], 'np': [0, 0, 0, 1]}
        history_event_one_hot = [
            [event_type_mapping[event] if event in event_type_mapping else [0, 0, 0, 0] for event in history]
            for history in history_event_types
        ]
        history_event_one_hot = torch.tensor(history_event_one_hot, dtype=torch.float32).to(self.device)  # (B, history_cnts, 4)
        history_event_one_hot_reshaped = history_event_one_hot.view(-1, self.event_dim)  # (B * history_cnt, event_dim)
        his_event_type_embedding = self.bn_event(self.event_linear(history_event_one_hot_reshaped))
        his_event_type_embedding = his_event_type_embedding.view(batch_size, history_cnt, -1)
        
        his_combined_embedding = torch.cat((his_image_embedding, his_coord_embedding, his_event_type_embedding), dim=-1)
        
        # 权重
        history_days = torch.tensor([[((datetime.strptime(now_info[i]['timestamp'], '%Y-%m-%d %H:%M:%S') - datetime.strptime(hist['timestamp'], '%Y-%m-%d %H:%M:%S')).days if hist['timestamp'] is not None else 0) for hist in history] for i, history in enumerate(history_info)], dtype=torch.float32).to(self.device)
        quality_indexes = torch.tensor([[hist['image_quality_index'] if hist['image_quality_index'] is not None else 0.0 for hist in history] for history in history_info], dtype=torch.float32).to(self.device)
        weights = self.compute_weight(history_days, quality_indexes)  # (B, max_history)
        weights = weights.unsqueeze(-1)  # (B, max_history, 1)
        
        # 输入到 Transformer Encoder 中
        his_combined_embedding_with_positional = self.positional_encoding_layer(his_combined_embedding)
        his_embedding_transformed = self.transformer_encoder(his_combined_embedding_with_positional.permute(1, 0, 2))  # 需要 (seq_len, batch_size, embed_dim)
        his_embedding_transformed = his_embedding_transformed.permute(1, 0, 2)  # 转换回来 (batch_size, seq_len, embed_dim)
        exp_weights = weights.expand_as(his_embedding_transformed)

        # No Weighting
        # averaged_embedding = his_embedding_transformed.mean(dim=1)

        # 对历史特征加权求和，得到一个加权后的历史特征 (B, total_dim)
        weighted_his_embedding = torch.sum(exp_weights * his_embedding_transformed, dim=1)

        # MLP + 后融合
        # 历史预测
        x_history = F.relu(self.history_fc1(weighted_his_embedding))
        x_history = F.relu(self.history_fc2(x_history))
        x_history = self.history_fc3(x_history)  # 输出形状为 (B, 1)

        output = x_history

        return output

    def get_img_feature(self, image_path, bbox):

        img_features = feature_extractor.extract_img_features(image_path, bbox)

        return img_features
    
    def compute_weight(self, timestamps, quality_indexes):
        """
        计算历史信息的权重。
        :param timestamps: 历史时间差列表
        :param quality_indexes: 历史图像质量指数列表
        :return: 计算出的权重列表
        """
        timestamps = torch.tensor(timestamps, dtype=torch.float32)
        quality_indexes = torch.tensor(quality_indexes, dtype=torch.float32)

        timestamps = torch.nan_to_num(timestamps, nan=0.0)
        quality_indexes = torch.nan_to_num(quality_indexes, nan=0.0)

        # 计算权重
        max_brisque_value = 75
        inverted_quality = max_brisque_value - quality_indexes
        weights = (inverted_quality) / (timestamps + 1)  # 避免除以0

        # 权重归一化
        row_sums = weights.sum(dim=1, keepdim=True)  # 每一行的和
        weights = weights / row_sums  # 每一行归一化

        return weights

    def weighted_average(self, features, weights):
        # 确保权重是张量
        weights = torch.tensor(weights, dtype=torch.float32)

        # 如果 features 是一维张量（如坐标特征）
        if features.dim() == 1:
            weighted_sum = torch.sum(weights * features)  # 逐元素相乘
        else:
            # 这里假设 features 的形状为 [N, D]，即 N 个样本，D 是特征维度
            if weights.shape[0] != features.shape[0]:
                raise ValueError("Weights shape must match the number of feature samples.")
            # 逐元素相乘并按列求和
            weighted_sum = torch.sum(features * weights.unsqueeze(1), dim=0)  # dim=0 按列求和

        return weighted_sum.unsqueeze(0)  # 返回形状为 [1, D] 的张量


if __name__ == "__main__":
    pass