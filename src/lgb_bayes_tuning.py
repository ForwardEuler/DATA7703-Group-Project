#!pip install bayesian-optimization

from bayes_opt import BayesianOptimization
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from f_ndarray import *

import warnings

warnings.filterwarnings('ignore')
plt.style.use('ggplot')

# from catboost import CatBoostRegressor
warnings.filterwarnings('ignore')

train = mload('../data_npz/train.npz')
test = mload('../data_npz/test.npz')

x_train = train.drop(['isDefault'], axis=1)
y_train = train['isDefault']
x_test = test.drop(['isDefault'], axis=1)
y_test = test['isDefault']

# Convert to lgb.Dataset
train_matrix_lgb = lgb.Dataset(x_train, label=y_train)
valid_matrix_lgb = lgb.Dataset(x_test, label=y_test)

"""
Bayes Tuning - Step 1&2: Define the objective function, Construct model
"""


def rf_cv_lgb(num_leaves, max_depth, bagging_fraction, feature_fraction, bagging_freq, min_data_in_leaf,
              min_child_weight, min_split_gain, reg_lambda, reg_alpha):
    # Construct model
    model_lgb = lgb.LGBMClassifier(
        boosting_type='gbdt',
        objective='binary',
        metric='auc',
        learning_rate=0.1,
        n_estimators=5000,
        num_leaves=int(num_leaves),
        max_depth=int(max_depth),
        bagging_fraction=round(bagging_fraction, 2),
        feature_fraction=round(feature_fraction, 2),
        bagging_freq=int(bagging_freq),
        min_data_in_leaf=int(min_data_in_leaf),
        min_child_weight=min_child_weight,
        min_split_gain=min_split_gain,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        n_jobs=8
    )

    val = cross_val_score(model_lgb, x_train, y_train,
                          cv=5, scoring='roc_auc').mean()

    return val


"""
Bayes Tuning - Step 3: Define the params to be optimized
"""
bayes_lgb = BayesianOptimization(
    rf_cv_lgb,
    {
        'num_leaves': (2, 200),
        'max_depth': (2, 20),
        'bagging_fraction': (0.5, 1.0),
        'feature_fraction': (0.5, 1.0),
        'bagging_freq': (0, 100),
        'min_data_in_leaf': (1, 100),
        'min_child_weight': (0, 10),
        'min_split_gain': (0.0, 1.0),
        'reg_alpha': (0.0, 10),
        'reg_lambda': (0.0, 10),
    }
)

"""
Start optimizing
"""
bayes_lgb.maximize(n_iter=5)

"""
result of optimization
"""
print(bayes_lgb.max)

"""
Modify learning rate to a relatively small one,
determine the optimal iteration via CV.
"""
best_params_lgb = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.01,
    'num_leaves': 18,
    'max_depth': 18,
    'min_data_in_leaf': 47,
    'min_child_weight': 6.4,
    'bagging_fraction': 0.98,
    'feature_fraction': 0.96,
    'bagging_freq': 67,
    'reg_lambda': 6,
    'reg_alpha': 6,
    'min_split_gain': 0.7,
    'nthread': 8,
    'seed': 42,
    'silent': True,
    'verbose': -1,
}

cv_result_lgb = lgb.cv(
    train_set=train_matrix_lgb,
    early_stopping_rounds=1000,
    num_boost_round=20000,
    nfold=5,
    stratified=True,
    shuffle=True,
    params=best_params_lgb,
    metrics='auc',
    seed=42
)

print('Number of iteration: {}'.format(len(cv_result_lgb['auc-mean'])))
print('AUC-cv of final optimal LGB model: {}'.format(max(cv_result_lgb['auc-mean'])))
# Number of iteration: 14269 --> num_boost_round
# AUC-cv of final optimal LGB model: 0.7315032037635779
