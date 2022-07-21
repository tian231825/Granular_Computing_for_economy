#!/usr/bin/env python

# encoding: utf-8
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/19 16:50
# @Author  : junruit
# @File    : Data_preprocessing.py
# @desc: PyCharm
'''
import torch
import pandas as pd


class Data(torch.nn.Module):
    def __init__(self, path):
        # 读取数据集
        self.data_full = pd.read_csv(path)
        # 得到表格的列名
        self.columns = list(self.data_full.columns)

    def data_preprocessing(self, id_list):
        # 被使用数据对应的列
        self.features = []
        for i in range(len(self.columns)):
            if i in id_list:
                self.features.append(self.columns[i])
        return self.features

    def return_coloumn(self):
        return self.features

    def return_use_data(self):
        use_data = self.data_full[self.features]
        return use_data