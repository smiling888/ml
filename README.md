# ml
# 数据清洗和预处理
 ## 分析数据
 1. 数据列数、数据类型()
```
head()
info()
数据类型
    train_drop.dtypes.sort_values()
将相同数据放一起
train_drop.select_dtypes(include='int64').head()
``` 
2. 补齐数据-检查是否有缺失值
 要么补齐、要么丢弃。
 补齐，使用平均值，最大值，随意值。
```
train.isnull().sum()[lambda x: x>0]
or
train.info()
```

## 特征提取
1. 分析每个属性，从中提取特征

- 从名称中提取 性别，
2. 分析特征与预测值的相关性
```
画出幸存与舱位的关系图
泰坦尼克
titanic['has_Cabin'].loc[~titanic.Cabin.isnull()]=1
titanic['has_Cabin'].loc[titanic.Cabin.isnull()]=0
pd.crosstab(titanic.has_Cabin[:len_train],train.Survived).plot.bar(stacked=True)
```
3.最后，将各种值中转为数字

- 分段：年龄分段映射为数字
- 字母：映射为0，1，2，3
```
titanic.Sex = titanic.Sex.map({'male':1,'female':0})
titanic.Cabin = titanic.Cabin.map({'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'U':7})
titanic.Embarked=titanic.Embarked.map({'C':0,'Q':1,'S':2})
titanic.Title=titanic.Title.map({'Mr':0,'Miss':1,'Mrs':2,'Master':3,'Rare':4})
```