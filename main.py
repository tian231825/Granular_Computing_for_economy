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
# example
feature_fetch_id = [0, 1, 3]
#
coloumn_name = origin_data.return_coloumn()
data = origin_data.return_use_data()
fcm = GranularComputing(data=data, max_iterion=100, labels=coloumn_name)
results, matrix_center, matrix = fcm(5)
