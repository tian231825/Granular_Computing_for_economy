# -*- encoding: utf-8 -*-
"""
@File    : main.py
@Time    : 2022/7/9 11:57
@Author  : junruitian
@Software: PyCharm
"""
import pandas as pd
from GrC import *


def data_preprocessing(path, id_list):
    data_full = pd.read_csv(path)
    # 得到表格的列名
    columns = list(data_full.columns)
    # 提取需要聚类的数据 去特定的列数id
    # features = columns[0:len(columns) - 1]
    features = []
    for i in range(len(columns)):
        if i in id_list:
            features.append(columns[i])
    data = data_full[features]
    # print(data)
    return features, data


data_file_path = "Iris.csv"
feature_fetch_id = [0, 1,  3]
coloumn_name, data = data_preprocessing(path=data_file_path, id_list=feature_fetch_id)
fcm = GranularComputing(data=data, max_iterion=100, labels=coloumn_name)
results, matrix_center, matrix = fcm(5)

