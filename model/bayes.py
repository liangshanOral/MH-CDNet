import os
import pandas as pd
import numpy as np
import random
import pickle 
from datetime import datetime, timedelta
import copy

# 贝叶斯融合计算
def cal_bayes(events, initial_prob=0.5, neutral_prob=0.5, half_time=90):
    """
    根据输入的事件数据，计算贝叶斯融合概率，并考虑时间衰减
    """
    updated_events = copy.deepcopy(events)
    # Initialize prior probability
    P_H = initial_prob
    prev_prob = updated_events[0]['probability']

    for i in range(1, len(updated_events)):

        event = updated_events[i]
        event_type = event['event_type']
        prob = event['probability']
        time_diff = event['time']

        decay_factor = np.power(0.5, time_diff / half_time)
   
        # Apply decay to the previous probability
        P_H = (prev_prob - neutral_prob) * decay_factor + neutral_prob     

        P_E_given_H = prob
        # Compute total probability of the event
        P_E = (P_E_given_H * P_H) + ((1 - P_E_given_H) * (1 - P_H))

        # Update belief
        if P_E != 0:
            P_H = (P_E_given_H * P_H) / P_E

        event['probability'] = P_H
        prev_prob = P_H

    return updated_events

def convert_to_bayes(csv_file, index_file, output_file):
    """
    把目前的数据形式转化成Bayes接受的。
    """
    index = pd.read_csv(index_file)
    data = pd.read_csv(csv_file, encoding='utf-8')

    predictions = []
    true_labels = []
    event_types = []
    trues = []

    for idx in index['Index']:  # 假设索引文件中有一列叫 'Index'
        events = []  # 用于存储贝叶斯融合事件的信息

        now_info = data.iloc[idx]  # 当前信息

        # event_type
        # if now_info['np'] == 1:
        #     event_type = "np"  # 代表 np
        # elif now_info['FP'] == 1:
        #     event_type = "fp"  # 代表 fp
        # elif now_info['tp'] == 1:
        #     event_type = "tp"  # 代表 tp
        # elif now_info['tn'] == 1:
        #     event_type = "tn" # 代表tn

        event_type = now_info['event_type']

        initial_prob = now_info['confidence']

        # 获取当前事件的时间戳
        now_timestamp = now_info['timestamp']
        now_timestamp = datetime.strptime(now_timestamp, '%Y-%m-%d %H:%M:%S')

        # 查找历史事件
        history_events = data[
            (data['class_num_id'] == now_info['class_num_id']) &
            (data['sign_id'] == now_info['sign_id']) &
            (data['visit_num_id'] < now_info['visit_num_id'])
        ].sort_values(by='visit_num_id', ascending=True)  # 按 visit_num_id 升序排列

        # 计算时间差并构建事件列表
        previous_timestamp = None
        for _, history_event in history_events.iterrows():
            # 提取历史事件的时间戳和概率
            history_timestamp = history_event['timestamp']
            history_timestamp = datetime.strptime(history_timestamp, '%Y-%m-%d %H:%M:%S')
            probability = history_event['confidence']  # 这里是你具体的列名

            # 计算时间差
            if previous_timestamp is not None:
                # 计算当前事件和上一个历史事件的时间差
                time_diff = (history_timestamp - previous_timestamp).days
            else:
                previous_timestamp = history_event['timestamp']
                time_diff = None

            # 将事件添加到列表中
            events.append({
                'event_type': event_type,
                'probability': probability,
                'time': time_diff
            })

            previous_timestamp = history_timestamp  # 更新为当前历史事件的时间戳

        now_time = (now_timestamp - previous_timestamp).days

        # 最后添加当前事件
        events.append({
            'event_type': event_type,
            'probability': initial_prob,
            'time': now_time  # 当前事件时间差为0
        })

        # 计算贝叶斯融合概率
        updated_events = cal_bayes(events)

        outputs = updated_events[-1]['probability']
        targets = events[-1]['probability']

        if outputs > 0.5 and (event_type == 'tp' or event_type == 'np'):
            true = 1 # 检测成功
        elif outputs < 0.5 and (event_type == 'fp' or event_type == 'tn'):
            true = 1 # 检测成功
        else:
            true = 0 #检测错误率

        # 保存预测、真实标签和 info
        predictions.append(outputs)
        true_labels.append(targets)
        event_types.append(event_type)
        trues.append(true)

    # 创建 DataFrame，将预测和真实标签配对
    results_df = pd.DataFrame({
        'Predicted Probability': predictions,
        'Actual Probability': true_labels,
        'Event Type': event_types,
        'If True': trues,
    })

    # 保存到 CSV 文件
    # results_df.to_csv(output_file, index=False)

def calculate_accuracy(csv_file):
    """
    读取贝叶斯计算结果的 CSV 文件，统计准确率并计算贝叶斯概率与实际概率的差异
    """
    # 读取 CSV 文件
    results_df = pd.read_csv(csv_file)

    # 按 'Event Type' 和 'If True'（true标签）进行分组
    accuracy_df = results_df.groupby(['Event Type', 'If True']).size().unstack(fill_value=0)

    # 计算每个事件类型的准确率
    accuracy_df['accuracy'] = accuracy_df[1] / (accuracy_df[0] + accuracy_df[1])

    # 显示准确率表
    print(accuracy_df)

def calculate_accuracy2(file_path):
    # 读取 CSV 文件
    results_df = pd.read_csv(file_path)

    # 确保数据为浮点数
    results_df['Predicted Label'] = pd.to_numeric(results_df['Predicted Probability'], errors='coerce')
    results_df['True Labels'] = pd.to_numeric(results_df['Actual Probability'], errors='coerce')

    # 设定阈值
    threshold = 0.5

    # 预测标签
    results_df['Predicted Label'] = (results_df['Predicted Probability'] > threshold).astype(int)

    # 真实标签
    results_df['True Label'] = (results_df['True Labels'] > threshold).astype(int)

    # 统计指标
    TP = ((results_df['Predicted Label'] == 1) & (results_df['True Label'] == 1)).sum()  # 真阳性
    TN = ((results_df['Predicted Label'] == 0) & (results_df['True Label'] == 0)).sum()  # 真阴性
    FP = ((results_df['Predicted Label'] == 1) & (results_df['True Label'] == 0)).sum()  # 假阳性
    FN = ((results_df['Predicted Label'] == 0) & (results_df['True Label'] == 1)).sum()  # 假阴性

    # 计算准确率
    accuracy = (TP + TN) / len(results_df)

    # 输出结果
    print(f"True Positives (TP): {TP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")
    print(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":

    index_file_path = 'data/final_val_indices.csv'
    # index_file_path = 'data/pne_10_indices.csv'
    csv_file_path = 'data/final_traffic_sign_data.csv'
    output_file = 'data/test_results_bayes.csv'
    convert_to_bayes(csv_file_path, index_file_path, output_file)
    calculate_accuracy(output_file)