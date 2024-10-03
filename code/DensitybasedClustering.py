"""
 @Author: Nut
 @FileName: DensitybasedClustering.py
 @DateTime: 2024/10/3 23:13
 @SoftWare: PyCharm
"""

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

# DBSCAN聚类
# epses = [i*0.005 for i in range(1,40)]
# min_sampleses = [3,4,5]
# epses = [0.065]
# min_sampleses = [4]
epses = [0.04]
min_sampleses = [4]

for eps in epses:
    for min_samples in min_sampleses:
        db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')  # 初始化DBSCAN算法
        db.fit(distance_matrix)  # 进行聚类
        cluster_labels = db.labels_  # 获取聚类标签

        # 计算并输出被正确聚类的样本比例
        accuracy = calculate_cluster_accuracy(labels, cluster_labels)

        # 计算异常值比率
        outlier_ratio = np.sum(cluster_labels == -1) / len(cluster_labels)  # 计算异常值（噪声点）比率

        if outlier_ratio < 0.6 and accuracy > 0.8:
            #         if outlier_ratio < 0.7 and accuracy > 0.5:
            print(f"eps：{eps}, min_pts：{min_samples}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"In Cluster Ratio: {1 - outlier_ratio:.4f}")  # 打印异常值比率

            accuracy, precision, recall, f1 = evaluate_clustering(labels, cluster_labels)
            print(f"Accuracy: {accuracy}")
            print(f"Weighted Precision: {precision}")
            print(f"Weighted Recall: {recall}")
            print(f"Weighted F1: {f1}")

            # 绘制聚类结果并保存图像
            plt.figure(figsize=(10, 6))  # 创建一个10x6的图像
            plt.scatter(range(len(cluster_labels)), cluster_labels, c=cluster_labels, cmap='viridis')  # 绘制聚类结果散点图
            plt.xlabel('Index')  # 设置X轴标签
            plt.ylabel('Cluster Label')  # 设置Y轴标签
            plt.title('DBSCAN Clustering Results')  # 设置图像标题

            # 添加网格线并指定网格线位置
            #             plt.xticks([0, 201, 481, 802, 864, 906])  # 设置X轴刻度线位置
            plt.xticks([0, 201, 481, 737, 841, 1045])  # 设置X轴刻度线位置
            plt.grid(True, axis='x')  # 启用X轴上的网格线

            output_path = "dbscan_clustering_results.png"  # 指定输出图像路径
            plt.savefig(output_path)  # 保存图像
            plt.show()  # 显示图像

            print(f"picture saved: {output_path}")  # 打印图像保存路径





