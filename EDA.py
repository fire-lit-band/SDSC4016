# read data, and summary each column

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 读取train.csv和test.csv文件
train_data = pd.read_csv('./data/train.csv',index_col=0)
# 在train.csv里面分train和test

# 2. 数据预处理
# 假设train.csv中，最后一列是标签列，其他列是特征列
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]
# 选出y_train==1的行
X_train = X_train[y_train == 0]

# 3. 数据探索
# 数据的维度
print(X_train.shape)
# 数据的前5行
print(X_train.head())
# 数据的统计信息
print(X_train.describe())
# 数据的缺失值
print(X_train.isnull().sum())
# 数据的标签分布
print(y_train.value_counts())
# 数据的相关性
corr = X_train.corr()

# 创建热力图
plt.figure(figsize=(8, 6))
sns.heatmap(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()
# 数据的分布(一列数据一个图）

# for column in X_train.columns[21:]:
#     plt.figure()
#     plt.hist(X_train[column], bins=100)
#     plt.title(f'Histogram of {column}')
#     plt.show()

# plt.show()
# 数据的箱线图
# X_train.boxplot()
# plt.show()
from sklearn.preprocessing import StandardScaler
print(X_train.columns[1])
scaler = StandardScaler()

# 使用fit_transform方法对数据进行标准化

for column in X_train.columns:
    # print(column)
    # plt.figure()
    # scaled_data = scaler.fit_transform(np.array(X_train[column]).reshape(-1, 1))
    # plt.boxplot(np.log(scaled_data))
    # plt.title(f'Boxplot of {column}')
    # plt.xticks([1], [column])
    # plt.show()
    print("mean:",X_train[column].mean())
    print("std",X_train[column].std())

plt.show()

# for column in X_train.columns:
#     plt.figure()
#     X_train[column].plot(kind='density', title=f'Density plot of {column}')
#     plt.show()