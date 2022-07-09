# -*- encoding: utf-8 -*-
"""
@File    : GrC.py
@Time    : 2022/7/9 11:17
@Author  : junruitian
@Software: PyCharm
"""
import random
import numpy as np
import torch
import operator
import math
from numpy import *


class GranularComputing(torch.nn.Module):
    def __init__(self, entity_num, max_iterion, labels):
        super(GranularComputing, self).__init__()
        self.entity_num = entity_num
        self.max_iterion = max_iterion
        # 模糊参数
        self.m = 2.00
        # 阈值
        self.Epsilon = 0.00000001
        self.labels = labels

    # 初始化模糊矩阵（隶属度矩阵 U）
    # 用值在0，1间的随机数初始化隶属矩阵，得到c列的U，使其满足隶属度之和为1
    def initialize_matrix_U(self, entity_num, cluster):
        # 返回一个模糊矩阵的列表
        matrix_ret = []
        # 标准化
        for i in range(entity_num):
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
    def iteration(self, matrix, cluster, data,):
        # 最大迭代次数：MAX_ITER=100
        iter = 0
        while iter <= self.max_iterion:
            iter += 1
            # 计算聚类中心矩阵 matrix_center
            matrix_center = self.calculateCenter(matrix=matrix, cluster=cluster, data=data)
            # 更新模糊矩阵 matrix_new
            matrix = self.matrix_update(init_matrix=matrix, matrix_center=matrix_center, data=data, cluster=cluster)
            # 得到更新后的中心矩阵
            matrix_center_new = self.calculateCenter(matrix=matrix, cluster=cluster, data=data)
            # 如果matrix_center_new和matrix_center的距离小于阈值，迭代停止
            distance = 0
            for i in range(cluster):
                for j in range(len(self.labels) - 1):
                    distance = (matrix_center_new[i][j] - matrix_center[i][j]) ** 2 + distance
            if sqrt(distance) < self.Epsilon:
                break
        return matrix_center, matrix

    # 更新隶属度矩阵 U
    def matrix_update(self, init_matrix, matrix_center, cluster, data):
        # 2/(m-1)
        p = float(2 / (self.m - 1))
        for i in range(self.entity_num):
            # 取出文件中的每一行数据
            x = list(data.iloc[i])
            # 求dij
            distances = [np.linalg.norm(list(map(operator.sub, x, matrix_center[j]))) for j in range(cluster)]
            for j in range(cluster):
                # 分母
                den = sum([math.pow(float(distances[j] / distances[cluster]), p) for cluster in range(cluster)])
                init_matrix[i][j] = float(1 / den)
        return init_matrix

    # 计算中心矩阵 V
    def calculateCenter(self, matrix, cluster, data):
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
                data_point = list(data.iloc[i])
                # uij的m次方 乘以 xj
                prod = [xraised[i] * val for val in data_point]
                temp_num.append(prod)
            # 分子：上面的结果求和
            numerator = map(sum, zip(*temp_num))
            # 求聚类中心
            center = [z / denominator for z in numerator]
            print(center)
            matrix_center.append(center)
        return matrix_center

    def forward(self, data, cluster):
        # 初始化模糊矩阵 U
        initial_matrix = self.initialize_matrix_U(entity_num=self.entity_num, cluster=cluster)
        # 迭代
        V, U = self.iteration(initial_matrix, cluster, data)
        return V
