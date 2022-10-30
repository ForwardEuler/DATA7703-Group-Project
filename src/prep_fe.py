import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from f_ndarray import mload, msave
import time


def printf(f, *args):
    print(f % args, end='')


print("Importing data...")
st = time.time()
df = mload('raw.npz')
et = time.time()
printf('Import complete, time taken: %.3fs\n', et - st)

print("Preprocessing..")
st = time.time()
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
df['dti_cal'] = df['loanAmnt'] / df['annualIncome']
df['ot_ratio'] = df['openAcc'] / df['totalAcc']
df['time_ratio'] = np.zeros(len(df))

for year in np.arange(2012, 2019, 1):
    tr, nz = np.empty(5), np.empty(4)
    for n, month in enumerate(np.arange(0, 13, 3)):
        tr[n] = year * 100 + month
    for i in range(4):
        lt = np.array(df.index[(df['issueDate'] > tr[i]) & (df['issueDate'] <= tr[i + 1])])
        t = df.iloc[lt]
        nz[i] = np.count_nonzero(t['isDefault'].values)
        df.iloc[lt, -1] = nz[i] / (len(lt) + 0.001)
        # printf("%d-%.2d has default rate of %.4f\n", year, tr[i] - year * 100 + 1, nz[i] / (len(lt) + 0.001))

uni_regn = df['regionCode'].unique()
df['loc_ratio'] = np.zeros(len(df))
for r in uni_regn:
    i = np.array(df.index[df['regionCode'] == r])
    t = df.iloc[i]
    n = np.count_nonzero(t['isDefault'].values)
    df.iloc[i, -1] = n / (len(i) + 0.001)
    # print(n / (len(i) + 0.001))

df.drop(['regionCode'], axis=1, inplace=True)
# drop extreme
ti = np.array(df.index[df['loc_ratio'] > 0.45])
df.drop(axis=0, index=ti, inplace=True)
ti = np.array(df.index[(df['dti_cal'] > 1) | (df['dti'] > 60)])
df.drop(axis=0, index=ti, inplace=True)
ti = np.array(df.index[df['annualIncome'] > 77974 + 4 * 70892])
df.drop(axis=0, index=ti, inplace=True)
df.reset_index(drop=True, inplace=True)

minmax = MinMaxScaler()
data_std = pd.DataFrame(minmax.fit_transform(df.values), columns=df.columns)

"""# 2 Imbalance issue: under-sampling"""
train_data, test_data = train_test_split(data_std, test_size=0.25, random_state=42)
non_default = train_data[train_data['isDefault'] == 0]
default = train_data[train_data['isDefault'] == 1]

# under sampling
non_default = pd.DataFrame.sample(non_default, int(1 * len(default)))

# combine
train_data = pd.concat([default, non_default])

# reset index
train_data = train_data.sample(frac=1, random_state=114514).reset_index(drop=True)
test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)
et = time.time()
printf('Preprocessing complete, time taken: %.3fs\n', et - st)

print("Compressing and saving data..")
st = time.time()
msave('data_numeric', df)
msave('train', train_data)
msave('test', test_data)
et = time.time()
printf('Saving complete, time taken: %.3fs\n', et - st)
