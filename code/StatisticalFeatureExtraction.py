"""
 @Author: Nut
 @FileName: StatisticalFeatureExtraction.py
 @DateTime: 2024/10/3 23:47
 @SoftWare: PyCharm
"""

import pandas as pd
import numpy as np
import itertools


# 特征提取函数
def extract_features(up_seq, down_seq):
    # 计算上行和下行序列的总和
    up_sum = np.sum(up_seq)
    down_sum = np.sum(down_seq)

    # 判断哪个序列和更大，将大的作为下行，小的作为上行
    if up_sum > down_sum:
        up_seq, down_seq = down_seq, up_seq
        up_sum, down_sum = down_sum, up_sum

    # 计算上下行各自的特征
    up_len = len(up_seq)
    up_mean = np.mean(up_seq)
    up_max = np.max(up_seq)
    up_std = np.std(up_seq)
    up_var = np.var(up_seq)
    up_skew = pd.Series(up_seq).skew()
    up_kurt = pd.Series(up_seq).kurt()
    up_zero_ratio = np.sum(up_seq == 0) / up_len

    down_len = len(down_seq)
    down_mean = np.mean(down_seq)
    down_max = np.max(down_seq)
    down_std = np.std(down_seq)
    down_var = np.var(down_seq)
    down_skew = pd.Series(down_seq).skew()
    down_kurt = pd.Series(down_seq).kurt()
    down_zero_ratio = np.sum(down_seq == 0) / down_len

    up_len_log = np.log(up_len + 1)
    down_len_log = np.log(down_len + 1)

    mean_ratio = max(up_mean, down_mean) / max(1, min(up_mean, down_mean))
    sum_ratio = down_sum / max(1, up_sum)

    # 计算整体序列特征
    combined_seq = np.concatenate([up_seq, down_seq])
    combined_len = len(combined_seq)
    combined_mean = np.mean(combined_seq)
    combined_max = np.max(combined_seq)
    combined_std = np.std(combined_seq)
    combined_var = np.var(combined_seq)
    combined_skew = pd.Series(combined_seq).skew()
    combined_kurt = pd.Series(combined_seq).kurt()
    combined_mean_max_ratio = combined_mean / max(1, combined_max)

    # 新增特征：统计combined_seq中具有连续20个以上0的区间的个数
    zero_intervals_count = np.sum([len(list(g)) >= 20 for k, g in itertools.groupby(combined_seq) if k == 0])

    features = [
        up_len_log, up_mean, up_max, up_std, up_var, up_skew, up_kurt,
        down_len_log, down_mean, down_max, down_std, down_var, down_skew, down_kurt,
        mean_ratio, up_zero_ratio, down_zero_ratio, up_sum, down_sum, sum_ratio,
        combined_len, combined_mean, combined_max, combined_std, combined_var,
        combined_skew, combined_kurt, combined_mean_max_ratio, zero_intervals_count
    ]

    return features


# 特征提取函数
def extract_features_4(up_seq, down_seq):
    # 计算上行和下行序列的总和
    up_sum = np.sum(up_seq)
    down_sum = np.sum(down_seq)

    # 判断哪个序列和更大，将大的作为下行，小的作为上行
    if up_sum > down_sum:
        up_seq, down_seq = down_seq, up_seq

    # 计算上下行各自的特征
    up_zero_ratio = np.sum(up_seq == 0) / len(up_seq)  # 上行序列中零值的比例
    down_mean = np.mean(down_seq)  # 下行序列的均值
    combined_len = len(up_seq) + len(down_seq)  # 上行和下行序列的总长度

    # 计算上下行均值的比例
    up_mean = np.mean(up_seq)
    mean_ratio = max(up_mean, down_mean) / max(1, min(up_mean, down_mean))

    # 将四个特征放入列表中
    features = [down_mean, combined_len, up_zero_ratio, mean_ratio]

    return features


# 逐行读取CSV文件并提取特征
def read_and_extract_features(down_file_path, up_file_path, label_flag):
    down_features = []
    labels = []

    with open(down_file_path, 'r', encoding='utf-8-sig') as down_file, open(up_file_path, 'r',
                                                                            encoding='utf-8-sig') as up_file:
        down_file_lines = down_file.readlines()
        up_file_lines = up_file.readlines()
        index = -1

        for down_line, up_line in zip(down_file_lines, up_file_lines):
            index += 1
            down_values = down_line.strip().split(',')
            up_values = up_line.strip().split(',')

            if (label_flag > -1):
                label = int(down_values[label_flag])
                labels.append(label)
            #             print([x.strip() for x in down_values[2:]])
            down_seq = np.array([float(x) for x in down_values[2:] if x.strip()])
            up_seq = np.array([float(x) for x in up_values[2:] if x.strip()])

            features = extract_features_4(up_seq, down_seq)
            down_features.append(features)

    return np.array(down_features), np.array(labels)

# 文件路径
down_file_path = '../data/ISCX-NonTor-2016/downlink_tor.csv'
up_file_path = '../data/ISCX-NonTor-2016/uplink_tor.csv'

feature_names = [
    'up_len_log', 'up_mean', 'up_max', 'up_std', 'up_var', 'up_skew', 'up_kurt',
    'down_len_log', 'down_mean', 'down_max', 'down_std', 'down_var', 'down_skew', 'down_kurt',
    'mean_ratio', 'up_zero_ratio', 'down_zero_ratio', 'up_sum', 'down_sum', 'sum_ratio',
    'combined_len', 'combined_mean', 'combined_max', 'combined_std', 'combined_var',
    'combined_skew', 'combined_kurt', 'combined_mean_max_ratio','zero_intervals_count'
]



# 提取特征
features, labels = read_and_extract_features(down_file_path, up_file_path, 0)
feature_names = ['down_mean','combined_len','up_zero_ratio','mean_ratio']

# 提取指定列和标签
selected_columns = ['down_mean', 'combined_len', 'up_zero_ratio', 'mean_ratio']
selected_indices = [feature_names.index(col) for col in selected_columns]

# selected_features = features[:, selected_indices]
selected_features = features

# if(labels):
selected_data = np.column_stack((selected_features, labels))
# 创建 DataFrame 并保存为 CSV 文件
part_df = pd.DataFrame(selected_data, columns=selected_columns + ['label'])
all_df = pd.DataFrame(np.column_stack((features, labels)), columns=feature_names + ['label'])

print(part_df.shape)
print(all_df.shape)

part_feature_file = '../data/ISCX-NonTor-2016/part_feature_tor.csv'
part_df.to_csv(part_feature_file, index=False)

print("选择的特征值已保存到{}文件中。".format(part_feature_file))

# print(selected_data[0])

all_feature_file = '../data/ISCX-NonTor-2016/all_feature_tor.csv'
all_df.to_csv(all_feature_file, index=False)

print("选择的特征值已保存到{}文件中。".format(all_feature_file))



