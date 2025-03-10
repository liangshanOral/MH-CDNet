
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import ast

class TrafficSignDataset(Dataset):
    def __init__(self, index_file, csv_file, map_file, max_history=4):
        """
        初始化数据集。
        :param index_file: 不同数据集的index
        :param csv_file: 总参考数据集 CSV 文件路径。
        :param max_history: 最多考虑的历史重访次数。
        :param map_file: 路牌地图文件
        """
        self.index = pd.read_csv(index_file)
        self.data = pd.read_csv(csv_file, encoding='utf-8')
        self.map = pd.read_csv(map_file)
        self.max_history = max_history

    def __len__(self):
        """返回数据集大小。"""
        return len(self.index)

    def __getitem__(self, idx):
        """获取指定索引的数据。"""
        # 获取当前重访的信息
        index = self.index.iloc[idx].values[0]
        now_info = self.data.iloc[index]
    
        # 查找信息
        class_id = now_info['class_num_id']
        sign_id = now_info['sign_id']
        visit_id = now_info['visit_num_id']

        now_info['coordinates'] = ast.literal_eval(now_info['coordinates'])  # 解析字符串为列表
        now_info['bbox'] = ast.literal_eval(now_info['bbox'])  # 解析字符串为列表

        # 根据 class_id、sign_id 和 visit_num 获取历史数据
        history_info = self.get_history_data(class_id, sign_id, visit_id)

        # 查找对应的高精地图信息
        map_info = self.map[(self.map['class_num_id'] == class_id) & (self.map['sign_id'] == sign_id)].iloc[0]
        map_info['coordinates'] = ast.literal_eval(map_info['coordinates'])  # 解析字符串为列表

        # 返回现在的信息和历史信息
        return now_info, history_info, map_info

    def get_history_data(self, class_id, sign_id, visit_id):
        """获取历史重访的信息，最多返回 max_history 次。"""
        # 过滤数据
        history = self.data[
            (self.data['class_num_id'] == class_id) & 
            (self.data['sign_id'] == sign_id) &
            (self.data['visit_num_id'] < visit_id)
        ]

        # 按 visit_num_id 排序
        history = history.sort_values(by='visit_num_id', ascending=False)  # 降序排列
        # 获取最多 max_history 条记录
        history_records = history.head(self.max_history)
        
        # 解析历史记录的坐标和边界框
        for i in range(len(history_records)):
            # 使用 .at 进行单个元素的赋值
            history_records.at[history_records.index[i], 'coordinates'] = ast.literal_eval(history_records.at[history_records.index[i], 'coordinates'])
            history_records.at[history_records.index[i], 'bbox'] = ast.literal_eval(history_records.at[history_records.index[i], 'bbox'])

        # 如果不足 max_history 条，添加空行
        while len(history_records) < self.max_history:
            empty_row = pd.Series([None] * len(history.columns), index=history.columns)
            history_records = pd.concat([history_records, empty_row.to_frame().T], ignore_index=True)

        return history_records.to_dict(orient='records')  # 转换为字典格式

def custom_collate_fn(batch):
    """
    自定义的 collate 函数，用于处理批量样本。

    :param batch: 批量样本列表，每个样本由当前信息和历史信息组成
    :return: 处理后的批量样本
    """
    now_info_list, history_info_list, map_info_list = zip(*batch)  # 解压每个样本的信息

    # 保持每个样本独立，而不是合并成一个整体
    batch_now_info = []
    batch_history_info = []
    batch_map_info = []

    for now_info in now_info_list:
        batch_now_info.append(now_info)

    for history_info in history_info_list:
        batch_history_info.append(history_info)

    for map_info in map_info_list:
        batch_map_info.append(map_info)

    return batch_now_info, batch_history_info, batch_map_info


if __name__ == "__main__":
    # 假设 indices.csv 和 traffic_sign_data.csv 的路径
    index_file_path = 'data/train_indices.csv'
    csv_file_path = 'data/traffic_sign_data.csv'

    # 创建数据集
    dataset = TrafficSignDataset(index_file_path, csv_file_path, max_history=4)

    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=custom_collate_fn)

    # 测试数据集
    def test_dataset(dataloader):
        for batch_idx, (now_info, history_info) in enumerate(dataloader):
            print(f'Batch {batch_idx + 1}:')
            print('Current Info:')
            print(now_info['visit_num_id'])  # 输出当前信息
            print('History Info:')
            #print(history_info)  # 输出历史信息
            
            # 检查历史信息的数量是否符合 max_history
            for history in history_info:
                print("history", history[3]['visit_num_id'])
                assert len(history) <= dataset.max_history, "History information exceeds max history limit"
            print(len(history_info))
            if batch_idx >= 1:  # 仅测试前两个批次
                break

    # 运行测试
    test_dataset(dataloader)
