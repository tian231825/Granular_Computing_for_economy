Granular Computing for Economy Enterprise

*粒计算企业评估*

**A. 应用文档**

项目中包含三个文件两个模块
+ main.py
+ Data_preprocessing.py
+ GrC.py

其中GrC.py中 GranularComputing 类为粒计算 Granular Computing 中 FCM 算法的实现。

Data_perprocessing.py 中作为接受数据并进行处理的模块

main.py 调用两个模块进行应用

**B. Data Input**

我们希望接受一种结构化的数据输入，如下表$T$所示：

| Annual Tax |  Annual Assets | Worker Num | Research Investment  | Enterprise Name |
| :----: | :----: | :----: | :----: | :----: |
|5.1|3.5|1.4|0.2| enterprise_1 |
|4.9|3.0|1.4|0.2| enterprise_2 |
|4.7|3.2|1.3|0.2| enterprise_3 |
|4.6|3.1|1.5|0.2| enterprise_4 |
|5.4|3.9|1.7|0.4| enterprise_5 |
| ... | ... | ... | ... | ... |
|4.9|3.1|1.5|0.1| enterprise_m |
|5.4|3.7|1.5|0.2| enterprise_n |
|4.8|3.4|1.6|0.2| enterprise_k |
|4.8|3.0|1.4|0.1| enterprise_p |

整个表格T包含 $(n+1) \times m$ 组数据, 其中$n$为被分析的企业数目, $m$为对应数据标签的数量
首行为$m$个数据label，之后$n \times m$个数据单元，数据之间由","分隔，如下所示。$T_{ij}$表示第$(i-1)$个企业第$j$个标签对应的值。

```
```

**B. Data PreProcessing**

*数据预处理*

在接受原始数据origin data后，以及由文字构成的label标签中，我们需要将文字信息转化为数学中的值的概念。以下文为例：

| id | 企业涉及（所属）行业 |
| :----: | :----: |
| 1 | 交通运输业 |
| 2 | 农业种植业 |
| 3 | 能源煤炭业 |
| 4 | 交通运输业 |
|... | ... |
| 100 | 新兴信息产业 |

在
其次，对于一些我们重点关注的label，我们增加Attention的机制

```
# 提取需要聚类的数据 去特定的列数id
# features = columns[0:len(columns) - 1]
```
***有一些数据标签对应的文本较为复杂不能简单的划分和归一化，需要专门应对特定的数据进行特定的处理，原则上最终的目标是将数据处理为一个结构化的structure-data***



**B. Code Implement**


对于数据处理， 对于attention所在的

**B. Code Comments**

B1