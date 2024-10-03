"""
 @Author: Nut
 @FileName: main.py
 @DateTime: 2024/10/3 23:28
 @SoftWare: PyCharm
"""

from sklearn.preprocessing import StandardScaler  # 导入StandardScaler用于特征标准化
from scipy.spatial.distance import euclidean  # 导入欧式距离计算方法
from collections import defaultdict
import numpy as np  # 导入NumPy库，用于数值计算
import pandas as pd  # 导入Pandas库，用于数据处理
from sklearn.cluster import DBSCAN  # 导入DBSCAN算法
from collections import Counter  # 导入Counter，用于统计
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于绘图
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def calculate_cluster_accuracy(true_labels, cluster_labels):
    unique_clusters = set(cluster_labels) - {-1}  # 获取所有非噪声点的聚类标签
    correct_count = 0  # 初始化正确聚类的样本计数
    total_count = 0  # 初始化总聚类的样本计数

    for cluster in unique_clusters:  # 遍历每个聚类标签
        cluster_indices = [i for i, x in enumerate(cluster_labels) if x == cluster]  # 获取属于当前聚类的所有数据点的索引
        cluster_true_labels = [true_labels[i] for i in cluster_indices]  # 获取当前聚类的所有真实标签
        most_common_label, count = Counter(cluster_true_labels).most_common(1)[0]  # 统计当前聚类中最常见的标签

        correct_count += count  # 统计正确聚类的样本数
        total_count += len(cluster_indices)  # 统计总的聚类样本数

    if total_count == 0:  # 检查是否有非噪声点的聚类
        return 0  # 如果没有，返回0

    accuracy = correct_count / total_count  # 计算准确率
    return accuracy


def evaluate_clustering(true_labels, cluster_labels):
    # 第一步：基于多数投票获得最终标签
    unique_clusters = set(cluster_labels) - {-1}  # 获取所有非噪声点的聚类标签
    final_labels = [-1] * len(true_labels)  # 初始化最终标签数组

    for cluster in unique_clusters:  # 遍历每个聚类标签
        cluster_indices = [i for i, x in enumerate(cluster_labels) if x == cluster]  # 获取属于当前聚类的所有数据点的索引
        cluster_true_labels = [true_labels[i] for i in cluster_indices]  # 获取当前聚类的所有真实标签

        # 统计当前聚类中最常见的标签
        most_common_label, _ = Counter(cluster_true_labels).most_common(1)[0]

        # 将当前聚类中的所有样本的标签设为最常见标签
        for i in cluster_indices:
            final_labels[i] = most_common_label

    # 第二步：筛选掉 final_labels 中值为 -1 的元素
    valid_indices = [i for i, label in enumerate(final_labels) if label != -1]
    filtered_true_labels = [true_labels[i] for i in valid_indices]
    filtered_final_labels = [final_labels[i] for i in valid_indices]

    # 第三步：计算加权指标
    accuracy = accuracy_score(filtered_true_labels, filtered_final_labels)
    weighted_precision = precision_score(filtered_true_labels, filtered_final_labels, average='weighted',
                                         zero_division=0)
    weighted_recall = recall_score(filtered_true_labels, filtered_final_labels, average='weighted', zero_division=0)
    weighted_f1 = f1_score(filtered_true_labels, filtered_final_labels, average='weighted', zero_division=0)

    return accuracy, weighted_precision, weighted_recall, weighted_f1

# 示例标签数组
labels = np.array(['audio'] * 201 + ['file-trans'] * 280 + ['video'] * 256 + ['voip'] * 104 + ['unknown'] * 204)
distance_matrix = np.load("../data/DatasetA/shape_based_distance_A.npy")

#DBSCAN聚类
epses = [0.04]
min_sampleses = [4]

for eps in epses:
    for min_samples in min_sampleses:
        db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')  # 初始化DBSCAN算法
        db.fit(distance_matrix)  # 进行聚类
        cluster_labels = db.labels_  # 获取聚类标签

# 读取特征文件
feature_df = pd.read_csv('../data/DatasetA/part_feature.csv')
features = feature_df.iloc[:, :-1].values  # 提取特征列
true_labels = feature_df.iloc[:, -1].values  # 提取标签列
# true_labels = labels_cleaned

# 对特征进行标准化
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 计算每个簇的特征均值向量
unique_clusters = set(cluster_labels) - {-1}  # 获取所有非噪声点的聚类标签
cluster_feature_means = {}
for cluster in unique_clusters:
    cluster_indices = [i for i, x in enumerate(cluster_labels) if x == cluster]
    cluster_features = features_scaled[cluster_indices]
    cluster_feature_means[cluster] = np.mean(cluster_features, axis=0)

# 将异常样本分配到最近的簇
new_cluster_labels = cluster_labels.copy()
# thresholds = [0.05 * i for i in range(10,40)]
thresholds = [1.15]

correct_counts = []
error_counts = []

anomaly_counts = []

accuraccy_list = []
anomaly_list = []

for d in thresholds:
    error_list = []
    # 正确分配和错误分配计数
    right, wrong, unknown = 0, 0, 0
    print("distance threshold{}".format(d))
    # 分配
    for i, label in enumerate(cluster_labels):
        # 算距离
        if label == -1:  # 如果是异常样本
            min_distance = float('inf')
            closest_cluster = -1
            for cluster, mean_vector in cluster_feature_means.items():
                distance = euclidean(features_scaled[i], mean_vector)
                if distance < min_distance:
                    min_distance = distance
                    closest_cluster = cluster

            # 可分
            if min_distance <= d:
                new_cluster_labels[i] = closest_cluster  # 重新分配到最近的簇

                # 输出分错的情况
                if (i < 201 and closest_cluster not in [0, 1, 2]  # audio
                        or 201 <= i < 481 and closest_cluster not in [3, 4, 5, 6, 7, 8]  # file-trans
                        or 481 <= i < 737 and closest_cluster not in [9, 10, 11, 12, 13, 14]  # video
                        or 802 <= i < 799 and closest_cluster not in [15, 16, 17, 18]  # voip
                        or i >= 841):
                    wrong += 1
                    error_list.append(i)
                #                     print("异常样本{}被错误分到簇{}中，特征是{}".format(i,closest_cluster,features_scaled[i]))
                # 输出分对的情况
                else:
                    right += 1
            #                     print("异常样本{}分类正确".format(i))
            # 异常
            else:
                new_cluster_labels[i] = -1  # 保持为异常样本
                unknown += 1
    #                 print("异常样本{}仍是异常样本，距离为{}".format(i,min_distance))

    # 计算重新分配后的准确率和异常值比率
    new_accuracy = calculate_cluster_accuracy(true_labels, new_cluster_labels)
    new_outlier_ratio = np.sum(new_cluster_labels == -1) / len(new_cluster_labels)

    print(f"Accuracy after Assignment: {new_accuracy:.4f}")
    print(f"In cluster Ratio after Assignment: {1 - new_outlier_ratio:.4f}")
    #     print(error_list)

    accuraccy_list.append(new_accuracy)
    anomaly_list.append(1 - new_outlier_ratio)


accuracy, precision, recall, f1 = evaluate_clustering(true_labels, new_cluster_labels)
print(f"Accuracy: {accuracy}")
print(f"Weighted Precision: {precision}")
print(f"Weighted Recall: {recall}")
print(f"Weighted F1: {f1}")


# 绘制重新分配后的聚类结果并保存图像
plt.figure(figsize=(10, 6))
plt.scatter(range(len(new_cluster_labels)), new_cluster_labels, c=new_cluster_labels, cmap='viridis')
plt.xlabel('Index')
plt.ylabel('Cluster Label')
plt.title('DBSCAN Clustering Results After Reassignment')

plt.xticks([0, 201, 481, 737, 841])  # 设置X轴刻度线位置
plt.grid(True, axis='x')

output_path = "dbscan_clustering_results_after_reassignment.png"
plt.savefig(output_path)
plt.show()

print(f"picture saved {output_path}")
