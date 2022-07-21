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
        self.path = path
        # 读取数据集
        self.data_full = pd.read_csv(self.path)
        # 得到表格的列名
        self.columns = list(self.data_full.columns)
        self.path_new = self.path.rpartition("/")[0] + "new_generate_" + self.path.split("/")[-1]

        # initial
        self.features = []
        self.use_data = []

    def data_preprocessing(self, id_list):
        # 被使用数据对应的列
        self.features = []
        for i in range(len(self.columns)):
            if i in id_list:
                self.features.append(self.columns[i])
        return self.features

    def text_transfer_label(self, id):
        label = self.columns[id]
        data_id = self.data_full[label]
        # print(type(data_id))
        # print(data_id)
        table_gen = []
        new_id = []
        for i in data_id:
            '''
            此处建议加入对于"\n","\t",","等特殊字符的处理
            '''
            i = i.replace("\n", "")
            i = i.replace("\t", "")
            i = i.replace(",", "")
            if i not in table_gen:
                table_gen.append(i)
                new_id.append(len(table_gen) - 1)
            else:
                for t in range(0, len(table_gen)):
                    if i == table_gen[t]:
                        new_id.append(t)
                        break
        # print(table_gen)
        # print(new_id)
        # 追加处理好的新生成的数据列
        new_colomn_name = str(label) + "_id"
        self.data_full[new_colomn_name] = new_id
        # 写入新文件
        self.data_full.to_csv(self.path_new, index=False, sep=',')
        # print(self.data_full)

    # 将id列缺失属性补充为对应value
    def missing_complement_num(self, id, value):
        label = self.columns[id]
        # print(label)
        self.data_full[label] = self.data_full[label].fillna(value=value)
        # 写入新文件
        self.data_full.to_csv(self.path_new, index=False, sep=',')
        print(self.data_full)

    # 将id列所有缺失属性变为同一个value
    def missing_complement_text(self, id):
        label = self.columns[id]
        self.data_full = pd.get_dummies(self.data_full, columns=[label], dummy_na=True)
        # 写入新文件
        self.data_full.to_csv(self.path_new, index=False, sep=',')

    def return_column(self, id_list):
        return self.data_preprocessing(id_list)

    def return_use_data(self):
        self.use_data = self.data_full[self.features]
        # print(use_data)
        return self.use_data

    def return_num_entity(self):
        return self.use_data.shape[0]


