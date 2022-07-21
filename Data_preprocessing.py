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


    def data_preprocessing(self):
        pass