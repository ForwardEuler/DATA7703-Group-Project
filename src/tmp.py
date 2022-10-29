import pandas as pd
import numpy as np
from random import sample
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from f_ndarray import mload, msave
from sklearn.model_selection import train_test_split


# read data
raw_data = pd.read_csv('train.csv')
# test_data = pd.read_csv('testA.csv')

raw_data['issueDate'] = pd.to_datetime(raw_data['issueDate'], format='%Y-%m-%d')
start_date = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
raw_data['issueDate'] = raw_data['issueDate'].apply(lambda x: int((x - start_date) / np.timedelta64(1, 'M')))

data = raw_data.sample(frac=1, random_state=42).reset_index(drop=True)
train_data, test_data = train_test_split(raw_data, test_size=0.25, random_state=114514)

"""
delete unrelated variable first to improve the running speed
['id'], ['policyCode'], ['isDefault']
id为无关变量
policyCode全部为1，对label无影响
假设issueDate（贷款日期）与y是否违约无关，因此删除
subGrade与grade高度相关，只保留subGrade
"""
data.drop(['id', 'grade', 'policyCode'], axis=1, inplace=True)

"""# 2 Imbalance issue: under-sampling"""

non_default = train_data[train_data['isDefault'] == 0]
default = train_data[train_data['isDefault'] == 1]

# under sampling
non_default = pd.DataFrame.sample(non_default, len(default))

# combine
train_data = pd.concat([default, non_default])
print('train shape is ', train_data.shape)

# reset index
train_data.reset_index(drop=True, inplace=True)

"""# 3 category variable
对象型类别特征需要进行预处理

## 3.1 时间格式特征：
earliesCreditLine: 借款人最早报告的信用额度开立的月份 \\
earliesCreditLine，只保留年份
"""

# earliesCreditLine原格式
print(train_data['earliesCreditLine'].sample(5))
data['earliesCreditLine'] = data['earliesCreditLine'].apply(lambda s: int(s[-4:]))
