import pandas as pd
import numpy as np
from numba import njit, jit

from f_ndarray import mload, msave
import time


def printf(f, *args):
    print(f % args, end='')


print("Importing data...")
st = time.time()
raw_data = pd.read_csv('data.csv')
et = time.time()
printf('Import complete, time taken: %.3fs\n', et - st)
data = raw_data.sample(frac=1, random_state=42).reset_index(drop=True)

"""
delete unrelated variable first to improve the running speed
['id'], ['policyCode'], ['isDefault']
id为无关变量
policyCode全部为1，对label无影响
subGrade与grade高度相关，只保留subGrade
"""
print("Encoding..")
data.drop(['id', 'grade', 'policyCode', 'employmentTitle', 'postCode', 'title'], axis=1, inplace=True)
data.drop(['ficoRangeHigh', 'n3', 'n9', 'n10'], axis=1, inplace=True)
data['earliesCreditLine'] = data['earliesCreditLine'].apply(lambda s: s[-4:])

data['issueDate'] = pd.to_datetime(data['issueDate'], format='%Y-%m-%d')
df_year = data['issueDate'].dt.year
df_month = data['issueDate'].dt.month
df_day = data['issueDate'].dt.day
df_date = df_year * 100 + df_month
data['issueDate'] = df_date


def employment_length_2_int(s: str):
    return s.split()[0]


data['employmentLength'].fillna('-1', inplace=True)
data['employmentLength'].replace(to_replace='10+ years', value='10 years', inplace=True)
data['employmentLength'].replace('< 1 year', '0 years', inplace=True)
data['employmentLength'] = data['employmentLength'].apply(employment_length_2_int)
data['employmentLength'].replace(to_replace='-1', value=np.nan, inplace=True)


def fn_grade(x: str):
    grade_dict = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}
    g = list(x)
    mg = grade_dict[g[0]]
    sg = int(g[1]) / 10
    fg = mg - sg
    return fg


data['subGrade'] = data['subGrade'].apply(fn_grade)
print("Encoding complete")

print("Compressing and saving data..")
st = time.time()
msave('raw', data)
et = time.time()
printf('Saving complete, time taken: %.3fs\n', et - st)
