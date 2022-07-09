# -*- encoding: utf-8 -*-
"""
@File    : main.py
@Time    : 2022/7/9 11:57
@Author  : junruitian
@Software: PyCharm
"""
import pandas as pd
from GrC import *


data_full = pd.read_csv("Iris.csv")
# 得到表格的列名
columns = list(data_full.columns)
# 前四个列名是鸢尾花特征（最后一列是鸢尾花种类）
features = columns[0:len(columns) - 1]
# 提取需要聚类的数据（根据列名提取前四列）
data = data_full[features]
fcm = GranularComputing(entity_num=10, max_iterion=100, labels=columns)
result = fcm(data, 3)

