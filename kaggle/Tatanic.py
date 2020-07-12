# coding=utf-8
# %matplotlib inline
"""
简书地址：https://www.jianshu.com/p/e79a8c41cb1a
kaggle地址: https://www.kaggle.com/c/titanic/data?select=train.csv
泰坦尼克号 -问题：预测测试集中的这些乘客能否存活
"""

import pandas as pd
import numpy as np
import matplotlib as plt
import warnings

warnings.filterwarnings('ignore')

train = pd.read_csv(r'../dataset/tatanic/train.csv')
test = pd.read_csv(r'../dataset/tatanic/test.csv')
PassengerId = test['PassengerId']
all_data = pd.concat([train, test], ignore_index=True)

#pd.set_option('display.max_rows', None) # 展示所有行
pd.set_option('display.max_columns', None) # 展示所有列
# head 默认展示5行
print(train.head())
#数据描述，以列为单位，统计数量:平均值，最大值等
print(train.describe())
# 数据信息，类似表描述，占用空间大小
#数据行数是891，而 Age只有714, 列有的信息是空的
print(train.info())
# 统计Age值每个值数量
print(train["Age"].value_counts())
print(train["Survived"].value_counts())
print(train["PassengerId"].value_counts())

