#!pip install bayesian-optimization
from sklearn.metrics import confusion_matrix
import pandas as pd
from bayes_opt import BayesianOptimization
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import confusion_matrix, roc_auc_score, confusion_matrix

from f_ndarray import mload
from lib_bc import *

import shap
import seaborn as sns
import warnings
import pickle
warnings.filterwarnings('ignore')

# from catboost import CatBoostRegressor

train = mload('../data_npz/train.npz')
test = mload('../data_npz/test.npz')

x_train = train.drop(['isDefault'], axis=1)
y_train = train['isDefault']
x_test = test.drop(['isDefault'], axis=1)
y_test = test['isDefault']

X_train_split, X_val, y_train_split, y_val = train_test_split(
    x_train, y_train, test_size=0.25)
# Convert to lgb.Dataset
train_matrix = lgb.Dataset(X_train_split, label=y_train_split)
valid_matrix = lgb.Dataset(X_val, label=y_val)
# setting 5-folds cv
kf = KFold(n_splits=5, shuffle=True, random_state=42)

##############################################################
######################Default Setting (Not tuned, Gauss)######
##############################################################

lgb_guess_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'learning_rate': 0.1,
    'metric': 'auc',
    'seed': 42,
    'nthread': 8,
    'silent': True,
    'verbose': -1,
}

lgb1 = lgb.train(lgb_guess_params,
                 train_set=train_matrix, valid_sets=valid_matrix,
                 num_boost_round=1000,
                 verbose_eval=1000,
                 early_stopping_rounds=200)
"""
Training until validation scores don't improve for 200 rounds.
Early stopping, best iteration is:
[300]	valid_0's auc: 0.732759
"""
# pred & performance
pred_test_lgb1 = lgb1.predict(x_test, num_iteration=lgb1.best_iteration)
# ROC
plot_roc(test_y_true=y_test, test_y_proba=pred_test_lgb1,
         name='LightGBM(default)')
# Confusion Matrix of test set
pred_valid_lgb1 = lgb1.predict(X_val, num_iteration=lgb1.best_iteration)
optimal_cutoff(y_val, pred_valid_lgb1)
# Optimal threshold that maximizes KS to 0.34299999999999997 is: threshold = 0.49000000000000027

pred_test_lgb_clf = score_map(pred_test_lgb1, 0.49)

labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
categories = ['Non-Default', 'Default']
lgb_default_cf = confusion_matrix(pred_test_lgb_clf, y_test)
make_confusion_matrix(lgb_default_cf,
                      group_names=labels,
                      categories=categories,
                      figsize=(10, 8),
                      cmap='Blues',
                      title="Confusion Matrix of LightGBM(default)")

##############################################################
######################Bayes Optimization######################
##############################################################
train_matrix_lgb = lgb.Dataset(x_train, label=y_train)

"""
Bayesopt - Step 1: Define the objective function, Construct model
"""
# Initialize domain space & define objective function
def lgb_cv_hyper_param(num_leaves, max_depth, bagging_fraction, feature_fraction, bagging_freq, min_data_in_leaf,
              min_child_weight, min_split_gain, reg_lambda, reg_alpha):
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
Bayesopt - Step 2: Define the params to be optimized

define the lower and upper bounds for each function input
"""
pbounds_lgb = {
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
bayes_lgb = BayesianOptimization(
    f= lgb_cv_hyper_param,
    pbounds=pbounds_lgb,
    random_state = 42
)

"""
Bayesopt - Step 3: Start the optimization process
"""
bayes_lgb.maximize(n_iter=10)


#the optimized values of the params 
print(bayes_lgb.max)
'''
{'target': 0.7354636916720081,
 'params': {'bagging_fraction': 0.9853946231813058,
  'bagging_freq': 67.11592580700115,
  'feature_fraction': 0.9667333325583202,
  'max_depth': 18.649984627291616,
  'min_child_weight': 6.385545738189469,
  'min_data_in_leaf': 47.70167974752877,
  'min_split_gain': 0.709391288325203,
  'num_leaves': 17.641470258530983,
  'reg_alpha': 6.504284388669609,
  'reg_lambda': 5.682748134912002}}
'''

"""
Bayesopt Step 4:
Modify learning rate to a relatively small one,
determine the optimal iteration via 5-folds CV.
n_iter = 4579, with auc-cv = 0.7360895393355424
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
print('AUC-cv of the tuned lgbm: {}'.format(max(cv_result_lgb['auc-mean'])))
# Number of iteration: 4579 --> num_boost_round
# AUC-cv of the tuned lgbm: 0.7360895393355424

# Valid in terms of the tuned params
"""
Use 5-folds CV to eval lightgbm performance
"""
cv_scores = []
for i, (train_ind, valid_ind) in enumerate(kf.split(x_train, y_train)):
    print('************************************ {} ************************************'.format(str(i+1)))
    X_train_split, y_train_split, X_val, y_val =\
        x_train.iloc[train_ind], y_train[train_ind],\
        x_train.iloc[valid_ind], y_train[valid_ind]

    train_mat= lgb.Dataset(X_train_split, label=y_train_split)
    valid_mat = lgb.Dataset(X_val, label=y_val)

    params = {
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
    }

    model = lgb.train(params, train_set=train_mat, num_boost_round=4579,
                      valid_sets=valid_mat, verbose_eval=1000, early_stopping_rounds=200)
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)

    cv_scores.append(roc_auc_score(y_val, val_pred))
    print(cv_scores)

print("lgb_auc_cv_list:{}".format(cv_scores))
print("lgb_auc_cv_mean:{}".format(np.mean(cv_scores)))
print("lgb_auc_std:{}".format(np.std(cv_scores)))

"""
lgb_auc_cv_list:[0.733850816064392, 0.7323502984752738, 0.7361959229609193, 0.7382719166348773, 0.7374121292738248]
lgb_auc_cv_mean:0.7356162166818574
lgb_auc_std:0.0022089347985254273

************************************ 1 ************************************
Training until validation scores don't improve for 200 rounds.
[1000]	valid_0's auc: 0.731666
[2000]	valid_0's auc: 0.733241
[3000]	valid_0's auc: 0.733829
Early stopping, best iteration is:
[2905]	valid_0's auc: 0.733851
[0.733850816064392]
************************************ 2 ************************************
Training until validation scores don't improve for 200 rounds.
[1000]	valid_0's auc: 0.72934
[2000]	valid_0's auc: 0.731222
[3000]	valid_0's auc: 0.732077
[4000]	valid_0's auc: 0.73235
Early stopping, best iteration is:
[3866]	valid_0's auc: 0.73235
[0.733850816064392, 0.7323502984752738]
************************************ 3 ************************************
Training until validation scores don't improve for 200 rounds.
[1000]	valid_0's auc: 0.73392
[2000]	valid_0's auc: 0.735668
[3000]	valid_0's auc: 0.73609
Early stopping, best iteration is:
[3486]	valid_0's auc: 0.736196
[0.733850816064392, 0.7323502984752738, 0.7361959229609193]
************************************ 4 ************************************
Training until validation scores don't improve for 200 rounds.
[1000]	valid_0's auc: 0.735337
[2000]	valid_0's auc: 0.73709
[3000]	valid_0's auc: 0.737942
[4000]	valid_0's auc: 0.738252
Early stopping, best iteration is:
[4182]	valid_0's auc: 0.738272
[0.733850816064392, 0.7323502984752738, 0.7361959229609193, 0.7382719166348773]
************************************ 5 ************************************
Training until validation scores don't improve for 200 rounds.
[1000]	valid_0's auc: 0.734654
[2000]	valid_0's auc: 0.736413
[3000]	valid_0's auc: 0.737024
[4000]	valid_0's auc: 0.737411
Early stopping, best iteration is:
[3860]	valid_0's auc: 0.737412
"""
# from 5folds-cv, it can be found that model will stop when iter_num meets 4182. --> later set num_boost_round to be 4182

##############################################################
######################Bayes Optimization END##################
##############################################################

##############################################################
################lgb(tuned) performance wrt. test set##########
##############################################################

#X_train_split, X_val, y_train_split, y_val = train_test_split(
#    x_train, y_train, test_size=0.25)
#train_matrix = lgb.Dataset(X_train_split, label=y_train_split)
#valid_matrix = lgb.Dataset(X_val, label=y_val)

# Train the tuned lgb

optimal_params = {
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
}
lgb_tuned = lgb.train(optimal_params,
                      train_set=train_matrix, valid_sets=valid_matrix,
                      num_boost_round=4182,
                      verbose_eval=1000,
                      early_stopping_rounds=200)
"""
Training until validation scores don't improve for 200 rounds.
[1000]	valid_0's auc: 0.73303
[2000]	valid_0's auc: 0.734985
[3000]	valid_0's auc: 0.735736
[4000]	valid_0's auc: 0.736155
Did not meet early stopping. Best iteration is:
[4098]	valid_0's auc: 0.736171
"""
# slightly improved on valid set

# Evaluation of the tuned lgb
# lgb(tuned) pred & performance on test set
pred_test_lgb_optimal = lgb_tuned.predict(
    x_test, num_iteration=lgb_tuned.best_iteration)

# ROC Curve
plot_roc(test_y_true=y_test, test_y_proba=pred_test_lgb_optimal,
         name='LightGBM(tuned)')
# AUC: 0.735

# Confusion Matrix
pred_valid_lgb_optimal = lgb_tuned.predict(
    X_val, num_iteration=lgb_tuned.best_iteration)
optimal_cutoff(y_val, pred_valid_lgb_optimal)
# Optimal threshold that maximizes KS to 0.3380000000000001 is: threshold = 0.5000000000000002

pred_lgb_tuned_clf = score_map(pred_test_lgb_optimal, 0.50)
labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
categories = ['Non-Default', 'Default']
lgb_tuned_cf = confusion_matrix(pred_lgb_tuned_clf, y_test)
make_confusion_matrix(lgb_tuned_cf,
                      group_names=labels,
                      categories=categories,
                      figsize=(10, 8),
                      cmap='Blues',
                      title="Confusion Matrix of LightGBM(tuned)")

# lgb - feature importance
# gain
features = x_train.columns.tolist()
pd.DataFrame({'Value':lgb_tuned.feature_importance(importance_type='gain'),'Feature':features}).sort_values(by="Value",ascending=False).head(20)
"""
|index|Value|Feature|
|---|---|---|
|4|1147856\.8891467154|subGrade|
|37|85972\.20726251602|time\_ratio|
|1|84258\.66650794446|term|
|35|82838\.16376423836|dti\_cal|
|9|68351\.10783733428|issueDate|
|6|55052\.08881865442|homeOwnership|
|11|54328\.76665624976|dti|
|38|53706\.54316857457|loc\_ratio|
|2|53118\.118864282966|interestRate|
|17|48303\.875457927585|revolBal|
|13|45524\.21037787199|ficoRangeLow|
|25|36535\.230126962066|n2|
|36|31719\.599650144577|ot\_ratio|
|7|30487\.43523274362|annualIncome|
|34|27128\.966501414776|n14|
|0|24937\.58013306558|loanAmnt|
|3|24119\.07088418305|installment|
|22|23509\.252723902464|earliesCreditLine|
|18|21825\.0922216177|revolUtil|
|28|16675\.435999274254|n6|
"""
# split
features = x_train.columns.tolist()
pd.DataFrame({'Value':lgb_tuned.feature_importance(importance_type='split'),'Feature':features}).sort_values(by="Value",ascending=False).head(20)
"""
|index|Value|Feature|
|---|---|---|
|9|5011|issueDate|
|17|4982|revolBal|
|7|4040|annualIncome|
|11|4014|dti|
|35|3774|dti\_cal|
|18|3751|revolUtil|
|2|3416|interestRate|
|36|3176|ot\_ratio|
|38|2966|loc\_ratio|
|3|2892|installment|
|22|2865|earliesCreditLine|
|0|2776|loanAmnt|
|28|2360|n6|
|13|2135|ficoRangeLow|
|19|2129|totalAcc|
|30|1984|n8|
|25|1781|n2|
|37|1742|time\_ratio|
|4|1503|subGrade|
|5|1345|employmentLength|
"""

def plot_feature_importance(imp, feat, model):

  feature_importance = np.array(imp)
  feature_names = np.array(feat)

  fi_dict = {'feature_names':feature_names,'feature_importance':feature_importance}
  fi_df = pd.DataFrame(fi_dict)

  fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

  plt.figure(figsize=(10,8))
  sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])

  plt.title(model + ' FEATURE IMPORTANCE')
  plt.xlabel('FEATURE IMPORTANCE')
  plt.ylabel('FEATURE NAMES')


plot_feature_importance(lgb_tuned.feature_importance(
    importance_type='gain'), features, 'LIGHTGBM (GAIN)')

plot_feature_importance(lgb_tuned.feature_importance(
    importance_type='split'), features, 'LIGHTGBM (SPLIT)')


