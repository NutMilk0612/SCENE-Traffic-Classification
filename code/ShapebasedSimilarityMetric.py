import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# 定义SBD函数
def sbd(x, y):
    def norm_cross_correlation(x, y):
        # 确保x和y是浮点类型的数组
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
        np.set_printoptions(threshold=np.inf)
        m = len(x)
        n = len(y)
        max_cc = -np.inf
        x_norm = np.linalg.norm(x)
        if x_norm == 0:
            return max_cc
        for s in range(-n + 1, m):
            if s >= 0:
                if s >= len(y):
                    continue
                y_shifted = np.pad(y, (s, 0), 'constant')[:m]
                x_segment = x[:len(y_shifted)]
            else:
                if -s >= len(y):
                    continue
                y_shifted = np.pad(y, (0, -s), 'constant')[-s:m]
                x_segment = x[-s:m]
            min_len = min(len(x_segment), len(y_shifted))
            x_segment = x_segment[:min_len]
            y_shifted = y_shifted[:min_len]
            if len(x_segment) == 0 or len(y_shifted) == 0 or np.all(y_shifted == 0):
                continue
            cc = np.sum(x_segment * y_shifted)
            y_norm = np.linalg.norm(y_shifted)
            if y_norm == 0:
                continue
            ncc = cc / (x_norm * y_norm)
            max_cc = max(max_cc, ncc)
        return max_cc

    ncc = norm_cross_correlation(x, y)
    if ncc == -np.inf:
        sbd_value = 2
    else:
        sbd_value = 1 - ncc
    return max(sbd_value, 0)

# 读取CSV文件并处理数据
file_path = "baseline_0831.csv"
data = []
labels = []
second_column_info = []  # 存储原始第二列信息

with open(file_path, 'r') as file:
    for line in file:
        fields = line.strip().split(',')
        label = fields[0]
        second_col = fields[1]  # 获取原始第二列信息
        sequence = list(map(float, fields[2:]))  # 提取序列数据并转换为浮点数

        # 数据清洗：去除前面的NaN值
        sequence = np.array(sequence)
        sequence = sequence[np.argmax(~np.isnan(sequence)):]  # 去除前面的NaN值

        labels.append(label)
        second_column_info.append(second_col)
        data.append(sequence)

# 数据清洗：去除全0的序列
non_zero_indices = [i for i in range(len(data)) if not np.all(data[i] == 0)]
data = [data[i] for i in non_zero_indices]
labels = [labels[i] for i in non_zero_indices]
second_column_info = [second_column_info[i] for i in non_zero_indices]

# 计算SBD距离矩阵
n = len(data)
distance_matrix = np.zeros((n, n))

for i in range(n):
    print("正在处理：", i)
    for j in range(i + 1, n):
        distance = sbd(np.array(data[i]), np.array(data[j]))
        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance

# 处理inf值，将其替换为距离矩阵中的最大有限值
finite_distances = distance_matrix[np.isfinite(distance_matrix)]
max_finite_distance = np.max(finite_distances)
distance_matrix[np.isinf(distance_matrix)] = max_finite_distance

# 保存距离矩阵到文件
np.save("0831_distance.npy", distance_matrix)
print("距离矩阵已保存到 0831_distance.npy")

# 计算每个点与其4个近邻点之间的 SBD 的均值
k = 4
kdis_values = []

for i in range(n):
    print(f"当前进度：{i+1}/{n}")  # 报数
    # 获取该点的所有距离
    distances = distance_matrix[i]
    # 排除自身距离
    distances = np.delete(distances, i)
    # 获取最近的k个距离
    nearest_distances = np.partition(distances, k)[:k]
    kdis_value = np.mean(nearest_distances)
    if not np.isinf(kdis_value):  # 检查 kdis_value 是否为 inf
        kdis_values.append(kdis_value)
    print(f"第{i+1}个序列的4-dis值: {kdis_value}")  # 输出中间计算结果

# 按降序排列 kdis 值
kdis_values.sort(reverse=True)
print("所有4-dis值（降序排列）：", kdis_values)

# 绘制 kdis 曲线并保存
plt.figure(figsize=(10, 6))
plt.plot(kdis_values)
plt.xlabel('index')
plt.ylabel('4-dis')
plt.title('4-dis Curve SBD')
output_path = "kdis_curve_0831.png"
plt.savefig(output_path)
plt.show()

print(f"4-dis 曲线已保存到 {output_path}")
