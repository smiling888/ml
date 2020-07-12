# coding=UTF-8
import numpy as np
from sklearn.model_selection import KFold

# 9*3
a = np.arange(27).reshape(9, 3)
print(a)
b = np.arange(9).reshape(9, 1)
# n_split=3表示，当执行KFold的split函数后，数据集被分成三份，两份训练集和一份验证集。
kfold = KFold(n_splits=3, shuffle=True)
#index = kfold.split(X=a)

#print(list(index))
#print(type(index))
# index是一个生成器，每个元素是一个元组，元组里面有两个元素，第一个是训练集的索引，第二个是验证集的索引
index = kfold.split(X=a, y=b)
# 类似下标集合
for train_index, test_index in index:
    print("-------------------------------------------------")
    print(train_index)
    print(a[train_index]) #注意如果a是datafram类型就得用a.iloc[tain_index], 因为a[train_index]会被认为是访问列
    print(a[test_index])