import os
import pandas as pd
import numpy as np
import random
import pickle 
from datetime import datetime, timedelta
import copy

# Dempster-Shafer 证据融合计算
def dempster_combination(mass1, mass2):
    combined_mass = {'∃': 0.5, '∄': 0.5}
    combined_mass['∃'] = mass1['∃'] * mass2['∃']
    combined_mass['∄'] = mass1['∄'] * mass2['∄']
    K = mass1['∃'] * mass2['∄'] + mass1['∄'] * mass2['∃']
    if K:
        combined_mass['∃'] /= (1 - K)
        combined_mass['∄'] /= (1 - K)
    total = combined_mass['∃'] + combined_mass['∄']
    combined_mass['∃'] /= total
    combined_mass['∄'] /= total
    return combined_mass

def get_mass_function(event_type, probability):
    if event_type:
        return {'∃': probability, '∄': 1 - probability}
    else:
        return {'∃': 0.5, '∄': 0.5}

def accumulate_mass_functions(events, half_time=90):
    if not events:
        return {'∃': 0.5, '∄': 0.5}, []
    probabilities = []
    combined_mass = get_mass_function(events[0]['event_type'], events[0]['probability'])
    for event in events[1:]:
        mass_t = get_mass_function(event['event_type'], event['probability'])
        #对前次事件进行时间衰减
        time_diff = event['time']
        decay_factor = np.power(0.5, time_diff / half_time)
        combined_mass['∃'] *= decay_factor
        combined_mass['∄'] *= decay_factor
        combined_mass = dempster_combination(combined_mass, mass_t)
        probabilities.append(combined_mass['∃'])
    return combined_mass, probabilities

def convert_to_ds(csv_file, index_file, output_file):
    """
    将目前的数据形式转化为Dempster-Shafer方法需要的格式
    """
    index = pd.read_csv(index_file)
    data = pd.read_csv(csv_file, encoding='utf-8')

    predictions = []
    true_labels = []
    event_types = []
    trues = []

    for idx in index['Index']:  # 假设索引文件中有一列叫 'Index'
        events = []  # 用于存储事件信息

        now_info = data.iloc[idx]  # 当前信息

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
        ].sort_values(by='visit_num_id', ascending=True)

        previous_timestamp = None
        for _, history_event in history_events.iterrows():
            history_timestamp = history_event['timestamp']
            history_timestamp = datetime.strptime(history_timestamp, '%Y-%m-%d %H:%M:%S')
            probability = history_event['confidence']

            if previous_timestamp is not None:
                time_diff = (history_timestamp - previous_timestamp).days
            else:
                previous_timestamp = history_event['timestamp']
                time_diff = None

            events.append({
                'event_type': event_type,
                'probability': probability,
                'time': time_diff
            })

            previous_timestamp = history_timestamp

        now_time = (now_timestamp - previous_timestamp).days

        # 最后添加当前事件
        events.append({
            'event_type': event_type,
            'probability': initial_prob,
            'time': now_time  # 当前事件时间差为0
        })

        # 使用DS证据理论融合计算
        combined_mass, probabilities = accumulate_mass_functions(events)

        # 获取最终的信任度
        final_belief = probabilities[-1] 
        targets = events[-1]['probability']

        if final_belief > 0.5 and (event_type == 'tp' or event_type == 'np'):
            true = 1  # 检测成功
        elif final_belief < 0.5 and (event_type == 'fp' or event_type == 'tn'):
            true = 1  # 检测成功
        else:
            true = 0  # 检测错误率

        predictions.append(final_belief)
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

if __name__ == "__main__":

    index_file_path = 'data/final_val_indices.csv'
    # index_file_path = 'data/pne_10_indices.csv'
    csv_file_path = 'data/final_traffic_sign_data.csv'
    output_file = 'data/test_results_ds.csv'
    convert_to_ds(csv_file_path, index_file_path, output_file)
    calculate_accuracy(output_file)
