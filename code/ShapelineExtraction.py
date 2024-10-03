import os
import numpy as np
import csv
import pandas as pd

# 文件路径
data_up_file = "/home/humingqi/Classification_of_encrypted_traffic/0829code/data-up-sorted-1045.csv"
data_down_file = "/home/humingqi/Classification_of_encrypted_traffic/0829code/data-down-sorted-1045.csv"

# 类别字典
class_dic = {0: "audio", 1: "file-trans", 2: "video", 3: "video-call", 4: "unknown"}

# 初始化列表存储数据
baseline_list = [[], [], [], [], []]

# 标准化函数
def standardize_series(series):
    series = pd.Series(series)
    std = series.std()
    if std == 0:
        return [0 for _ in series.tolist()]  # 如果标准差为零，返回全0序列
    standardized_series = (series - series.mean()) / std
    return standardized_series.tolist()

# 移除偏离均值最大的5%数据并进行线性插值
def remove_and_interpolate(series):
    series = pd.Series(series)
    deviation = (series - series.mean()).abs()
    threshold = int(len(series) * 0.05)
    sorted_indices = deviation.sort_values(ascending=False).index
    to_remove_indices = sorted_indices[:threshold]
    remaining = series.drop(to_remove_indices)
    interpolated = remaining.interpolate(method='linear')
    return interpolated.tolist()

# 小滑动窗口的移动平均法
def moving_average(series):
    half_len = len(series) // 2
    zero_ratio_first_half = np.sum(series[:half_len] == 0) / half_len
    
    if zero_ratio_first_half > 0.5:
        window_size = max(2, int((len(series) / 2) ** 0.5))
    else:
        window_size = max(2, int((len(series) / 2) ** 0.5 * 2))
    
    smoothed = series.rolling(window=window_size).mean().dropna().tolist()
    return smoothed

# 去掉序列末尾的零
def trim_trailing_zeros(sequence):
    last_non_zero_index = len(sequence) - 1
    while last_non_zero_index >= 0 and sequence[last_non_zero_index] == 0:
        last_non_zero_index -= 1
    return sequence[:last_non_zero_index + 1]

# 逐行读取CSV文件
with open(data_up_file, 'r') as f_up, open(data_down_file, 'r') as f_down:
    reader_up = csv.reader(f_up)
    reader_down = csv.reader(f_down)

    for row_up, row_down in zip(reader_up, reader_down):
        class_id = int(row_up[0])
        file_name = row_up[1]
        print(file_name)
        
        # 过滤掉无法转换为整数的值
        packet_bytes_up = [int(x) for x in row_up[2:] if x.isdigit()]
        packet_bytes_down = [int(x) for x in row_down[2:] if x.isdigit()]

        avg_up = np.mean(packet_bytes_up)
        avg_down = np.mean(packet_bytes_down)
        if avg_up > avg_down:
            packet_bytes1, packet_bytes2 = packet_bytes_down, packet_bytes_up
        else:
            packet_bytes1, packet_bytes2 = packet_bytes_up, packet_bytes_down

        #合并
        combined_series = packet_bytes1 + packet_bytes2
        #标准化
        standardized_series = standardize_series(combined_series)
        #降噪
        baseline_series = remove_and_interpolate(pd.Series(standardized_series))
        #平滑
        smooth_baseline_series = moving_average(pd.Series(baseline_series))
        
        baseline_list[class_id].append([class_dic[class_id], file_name] + smooth_baseline_series)

# 将基线数据写入CSV
with open('baseline_0831.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for sub_list in baseline_list:
        for i in sub_list:
            trimmed = trim_trailing_zeros(i[2:])  # 只处理数据部分
            writer.writerow([i[0], i[1]] + trimmed)


print("基线数据已保存到 'baseline_0831.csv' .")
