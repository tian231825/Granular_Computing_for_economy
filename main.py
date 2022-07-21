# -*- encoding: utf-8 -*-
"""
@File    : main.py
@Time    : 2022/7/16 11:57
@Author  : junruitian
@Software: PyCharm
"""
import pandas as pd

from GrC import *
from Data_preprocessing import *

# initial 初始化
data_file_path = "Iris.csv"
origin_data = Data(path=data_file_path)

'''
data_preprocessing过程
这里对应origin_data 不同种类的问题进行预处理清洗，
选择不同的接口或在Data类中自制更多的接口
'''
origin_data.text_transfer_label(4)
# origin_data.missing_complement_num(6, 0)
# origin_data.missing_complement_text(7)

# # 选取列id 示例
feature_fetch_id = [0, 1, 3]
# #
column_name = origin_data.return_column(id_list=feature_fetch_id)
# print(column_name)
data = origin_data.return_use_data()
# print(data)
entity_num = origin_data.return_num_entity()
# print(entity_num)
fcm = GranularComputing(data=data, max_iterion=100, labels=column_name)
cluster = math.sqrt(entity_num)
results, matrix_center, matrix = fcm(5)
