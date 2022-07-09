# -*- encoding: utf-8 -*-
"""
@File    : GrC.py
@Time    : 2022/7/9 11:17
@Author  : junruitian
@Software: PyCharm
"""
import random
import time
import matplotlib.pyplot as plt
import torch
from numpy import *
import numpy as np
import operator
import math

class GranularComputing(torch.nn.Module):
    def __init__(self, data, max_iterion, labels):
        super(GranularComputing, self).__init__()
        self.data = data
        self.entity_num = len(data)
        self.max_iterion = max_iterion
        # 模糊参数
        self.m = 2.00
        # 阈值
        self.Epsilon = 0.00000001
        self.labels = labels

    # 初始化模糊矩阵（隶属度矩阵 U）
    # 用值在0，1间的随机数初始化隶属矩阵，得到c列的matrix，使其满足隶属度之和为1
    def initialize_matrix(self, cluster):
        # 返回一个模糊矩阵的列表
        matrix_ret = []
        # 标准化
        for i in range(self.entity_num):
            # 初始化，给与随机的隶属度
            random_list = [random.random() for i in range(cluster)]
            # print(random_list)
            # 标准化：值/每列的和
            summation = sum(random_list)
            # print(summation)
            temp_list = [x / summation for x in random_list]
            # print(temp_list)
            matrix_ret.append(temp_list)
        # print(type(matrix_ret))
        return matrix_ret

    # 迭代，最多迭代MAX_ITER次
    # 计算中心矩阵V——》更新隶属度矩阵U——》计算更新后的中心矩阵V_update，V_update和V的距离若小于阈值则停止
    def iteration(self, matrix, cluster):
        # 最大迭代次数：MAX_ITER=100
        iter = 0
        while iter <= self.max_iterion:
            iter += 1
            # 计算聚类中心矩阵 matrix_center
            matrix_center = self.calculateCenter(matrix=matrix, cluster=cluster)
            # 更新模糊矩阵 matrix_new
            matrix = self.matrix_update(init_matrix=matrix, matrix_center=matrix_center, cluster=cluster)
            # 得到更新后的中心矩阵
            matrix_center_new = self.calculateCenter(matrix=matrix, cluster=cluster)
            # 如果matrix_center_new和matrix_center的距离小于阈值，迭代停止
            distance = 0
            for i in range(cluster):
                for j in range(len(self.labels)):
                    distance = (matrix_center_new[i][j] - matrix_center[i][j]) ** 2 + distance
            if sqrt(distance) < self.Epsilon:
                break
        return matrix_center, matrix

    # 更新隶属度矩阵 U
    def matrix_update(self, init_matrix, matrix_center, cluster):
        # 2/(m-1)
        p = float(2 / (self.m - 1))
        for i in range(self.entity_num):
            # 取出文件中的每一行数据
            x = list(self.data.iloc[i])
            # 求dij
            distances = [np.linalg.norm(list(map(operator.sub, x, matrix_center[j]))) for j in range(cluster)]
            for j in range(cluster):
                # 分母
                den = sum([math.pow(float(distances[j] / distances[cluster]), p) for cluster in range(cluster)])
                init_matrix[i][j] = float(1 / den)
        return init_matrix

    # 计算中心矩阵 V
    def calculateCenter(self, matrix, cluster):
        # 转置
        matrix_tranpose = list(zip(*matrix))
        # 中心矩阵，列表
        matrix_center = []
        for j in range(cluster):
            x = matrix_tranpose[j]
            # uij的m次方，m为模糊参数
            xraised = [e ** self.m for e in x]
            # 分母
            denominator = sum(xraised)
            temp_num = []
            # 取出转置矩阵每列的实体数量个元素
            for i in range(self.entity_num):
                # 得到分子中的 xj
                data_point = list(self.data.iloc[i])
                # uij的m次方 乘以 xj
                prod = [xraised[i] * val for val in data_point]
                temp_num.append(prod)
            # 分子：上面的结果求和
            numerator = map(sum, zip(*temp_num))
            # 求聚类中心
            center = [z / denominator for z in numerator]
            # print(center)
            matrix_center.append(center)
        return matrix_center

    # 获得聚类结果（判断样本属于哪个类）
    def get_cluster_id(self, matrix):
        results = list()
        # for循环取出U矩阵的实体数量行数据
        for i in range(self.entity_num):
            # 此时每条数据有cluster个隶属度，取最大的那个，并返回index值，即该实体点归属对应的cluster类簇
            max_value, index = max((value, index) for (index, value) in enumerate(matrix[i]))
            # 以index对应的clusre_id作为结果
            results.append(index)
        return results

    # Xie-Beni聚类有效性
    def valid_xie_beni(self, membership_mat, center, cluster):
        membership_mat = np.array(membership_mat)
        center = np.array(center)
        data_array = np.array(self.data)
        sum_cluster_distance = 0
        min_cluster_center_distance = inf
        for i in range(cluster):
            for j in range(self.entity_num):
                sum_cluster_distance = sum_cluster_distance + membership_mat[j][i] ** 2 * sum(
                    power(data_array[j, :] - center[i, :], 2))  # 计算类一致性
        for i in range(cluster - 1):
            for j in range(i + 1, cluster):
                cluster_center_distance = sum(power(center[i, :] - center[j, :], 2))  # 计算类间距离
                if cluster_center_distance < min_cluster_center_distance:
                    min_cluster_center_distance = cluster_center_distance
        return sum_cluster_distance / (self.entity_num * min_cluster_center_distance)

    def result_plt_show(self, results, center):
        # matplotlib需要array类型的数据
        data_array = np.array(self.data)
        center = np.array(center)
        results = np.array(results)
        # example
        # 将DATA的第一列和第二列作为x、y轴绘图
        plt.xlim(4, 8)
        plt.ylim(1, 5)
        # 创建一个绘图窗口
        plt.figure(1)
        # 画散点图
        # 样本点   其中nonzero(results==0)为取出0这一类的下标
        plt.scatter(data_array[nonzero(results == 0), 0], data_array[nonzero(results == 0), 1], marker='o', color='r', label='0',
                    s=30)
        plt.scatter(data_array[nonzero(results == 1), 0], data_array[nonzero(results == 1), 1], marker='+', color='b', label='1',
                    s=30)
        plt.scatter(data_array[nonzero(results == 2), 0], data_array[nonzero(results == 2), 1], marker='*', color='g', label='2',
                    s=30)
        plt.scatter(data_array[nonzero(results == 3), 0], data_array[nonzero(results == 3), 1], marker='*', color='y',
                    label='3',
                    s=30)
        plt.scatter(data_array[nonzero(results == 4), 0], data_array[nonzero(results == 4), 1], marker='+', color='y',
                    label='3',
                    s=30)
        # 中心点
        plt.scatter(center[:, 0], center[:, 1], marker='x', color='m', s=50)
        plt.show()

    def forward(self, cluster):
        # 记录初始化时间
        start = time.time()
        # 初始化模糊矩阵 U
        initial_matrix = self.initialize_matrix(cluster=cluster)
        # 迭代
        matrix_center, matrix = self.iteration(matrix=initial_matrix, cluster=cluster)
        # 获得聚类结果
        results = self.get_cluster_id(matrix=matrix)
        print(results)
        # 打印聚类所用时长
        print("用时：{0} s".format(time.time() - start))
        valid_effect = self.valid_xie_beni(matrix, matrix_center, cluster)
        print("聚类有效性：", valid_effect)

        # 可视化结果
        self.result_plt_show(results, matrix_center)
        return results, matrix_center, matrix
