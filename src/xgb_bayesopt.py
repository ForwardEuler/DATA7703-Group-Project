#!pip install bayesian-optimization
#!pip install shap
from sklearn.metrics import confusion_matrix
import pandas as pd
from bayes_opt import BayesianOptimization
import xgboost as xgb
from sklearn.model_selection import cross_val_score, train_test_split, KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, confusion_matrix

from f_ndarray import mload
from lib_bc import *

import shap
import seaborn as sns 
import warnings
warnings.filterwarnings('ignore')

train = mload('../data_npz/train.npz')
test = mload('../data_npz/test.npz')

x_train = train.drop(['isDefault'], axis=1)
y_train = train['isDefault']
x_test = test.drop(['isDefault'], axis=1)
y_test = test['isDefault']

X_train_split, X_val, y_train_split, y_val = train_test_split(
    x_train, y_train, test_size=0.25)

# Convert to xgb.DMatrix
train_matrix = xgb.DMatrix(x_train, label=y_train)
train_matrix_xgb = xgb.DMatrix(X_train_split, label=y_train_split)
valid_matrix_xgb = xgb.DMatrix(X_val, label=y_val)

# setting 5-folds cv
kf = KFold(n_splits=5, shuffle=True, random_state=42)


##############################################################
######################Bayes Optimization######################
##############################################################


# define objective function


def xgb_cv_hyper_param(gamma, min_child_weight, max_depth, reg_alpha, reg_lambda, subsample, colsample_bytree, colsample_bylevel, scale_pos_weight):
    # https://xgboost.readthedocs.io/en/latest/parameter.html
    model_xgb = xgb.XGBClassifier(
        booster='gbtree',
        objective='binary:logistic',
        eval_metric = 'auc',
        learning_rate = 0.1, # later 
        gamma = int(gamma),
        min_child_weight = min_child_weight,
        max_depth = int(max_depth),
        reg_alpha = reg_alpha,
        reg_lambda = reg_lambda,
        subsample = round(subsample, 2),
        colsample_bytree = round(colsample_bytree, 2), 
        colsample_bylevel=round(colsample_bylevel, 2),
        scale_pos_weight=round(scale_pos_weight, 2),
        # A typical value to consider: sum(negative instances) / sum(positive instances), here 4/1
    )

    val = cross_val_score(model_xgb, x_train, y_train, cv=3, scoring='roc_auc').mean()

    return val

# Initialize domain space

"""
two ways to control overfitting in XGBoost
First way, directly control model complexity: max_depth, min_child_weight, gamma
Second, add randomness to make training robust to noise: subsample, colsample_bytree. Moreover, reduce stepsize `eta`, but increase `num_round` when do so.
"""
pbounds_xgb = {
    'gamma': (0,5),
    'min_child_weight': (0, 10),
    'max_depth': (3,10),
    'reg_alpha': (0.0, 10),
    'reg_lambda': (0.0, 10),
    'subsample': (0.5, 0.8),
    'colsample_bytree': (0.5, 0.8),
    'colsample_bylevel': (0.5, 0.8),
    'scale_pos_weight': (0.5, 4)
}

bayes_xgb = BayesianOptimization(
    f = xgb_cv_hyper_param, 
    pbounds=pbounds_xgb,
    random_state = 42
)

bayes_xgb.maximize(n_iter=10)
#the optimized values of the params
print(bayes_xgb.max)

best_params_xgb = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'gamma': 1,
    'min_child_weight': 1.5,
    'max_depth': 5,
    'lambda': 10,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'colsample_bylevel': 0.7,
    'eta': 0.04,
    'tree_method': 'exact',
    'seed': 42,
    'nthread': 36,
    "silent": True,
}

cv_result_xgb = xgb.cv(
    dtrain = train_matrix,
    early_stopping_rounds=1000,
    num_boost_round=20000,
    nfold=5,
    stratified=True,
    shuffle=True,
    params=best_params_xgb,
    metrics='auc',
    seed=42
)
print('Number of iteration: {}'.format(len(cv_result_xgb['auc-mean'])))
print('AUC-cv of the tuned xgb: {}'.format(max(cv_result_xgb['auc-mean'])))


# valid in terms of the tuned params
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(x_train))
xgb_pred = np.zeros(len(x_test))

for fold_, (trn_idx, val_idx) in enumerate(kf.split(x_train, y_train)):
    print("Fold {}".format(fold_ + 1))
    trn_data = xgb.DMatrix(x_train.iloc[trn_idx], y_train.iloc[trn_idx])
    val_data = xgb.DMatrix(x_train.iloc[val_idx], y_train.iloc[val_idx])

    xgb_tuned = xgb.train(params=best_params_xgb,
                    dtrain=trn_data,
                    num_boost_round=2000, 
                    evals=[(trn_data, 'train'), (val_data, 'valid')],
                    maximize=False,
                    early_stopping_rounds=100,
                    verbose_eval=100)

    oof[val_idx] = xgb_tuned.predict(xgb.DMatrix(
        x_train.iloc[val_idx]), ntree_limit=xgb_tuned.best_ntree_limit)

    xgb_pred += xgb_tuned.predict(xgb.DMatrix(x_test),
                                  ntree_limit=xgb_tuned.best_ntree_limit) / kf.n_splits


"""
xgb_auc_cv_list:
  tr: (0.764227, 0.769825, 0.768921, 0.771175, 0.771495)
  valid: (0.737024, 0.734345, 0.734861, 0.736507, 0.736739)
auc-mean:
  tr: 0.7691286
  valid: 0.7358952
auc-std:
  tr: 0.00262
  valid: 0.00108
"""

# ROC Curve
plot_roc(test_y_true=y_test, test_y_proba=xgb_pred,
         name='XGBoost(tuned)')
# AUC: 0.7366653703335521

# Confusion Matrix
pred_valid_xgb_optimal = xgb_tuned.predict(
    xgb.DMatrix(X_val), ntree_limit=xgb_tuned.best_ntree_limit)
optimal_cutoff(y_val, pred_valid_xgb_optimal)
# Optimal threshold that maximizes KS to 0.3380000000000001 is: threshold = 0.5000000000000002

pred_xgb_tuned_clf = score_map(pred_valid_xgb_optimal, 0.50)
labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
categories = ['Non-Default', 'Default']
xgb_tuned_cf = confusion_matrix(pred_xgb_tuned_clf, y_test)
make_confusion_matrix(xgb_tuned_cf,
                      group_names=labels,
                      categories=categories,
                      figsize=(10, 8),
                      cmap='Blues',
                      title="Confusion Matrix of XGBoost(tuned)")

fea_gain_dict = xgb_tuned.get_score(importance_type='gain')
dict(sorted(fea_gain_dict.items(), key=lambda item: item[1], reverse=True))
"""
{'subGrade': 96.43257063499004,
 'term': 55.08578808688459,
 'interestRate': 30.12050781280053,
 'homeOwnership': 17.07338658120476,
 'time_ratio': 13.047038039096565,
 'ficoRangeLow': 9.235817965189876,
 'n14': 8.140536846312395,
 'dti_cal': 7.887565093875001,
 'loc_ratio': 7.579223773607824,
 'n2': 7.252899082820822,
 'issueDate': 6.667482205436628,
 'dti': 5.6406691936498286,
 'loanAmnt': 5.2852679440583215,
 'revolBal': 5.032082204146566,
 'delinquency_2years': 4.794830918283579,
 'annualIncome': 4.775008807572266,
 'installment': 4.765455189776947,
 'verificationStatus': 4.7070553509142865,
 'ot_ratio': 4.6782632055974815,
 'pubRec': 4.4986178844968165,
 'n1': 4.386090556065488,
 'earliesCreditLine': 4.343902190864252,
 'purpose': 4.284049564121534,
 'applicationType': 4.1964504909722224,
 'n6': 4.042537713273692,
 'n0': 4.0204931611660495,
 'employmentLength': 3.969994794290431,
 'n7': 3.956920106601513,
 'revolUtil': 3.95650672647554,
 'pubRecBankruptcies': 3.818668241559633,
 'totalAcc': 3.7942842843533193,
 'n5': 3.7867724749983527,
 'n8': 3.709128868133059,
 'n4': 3.596990702453761,
 'initialListStatus': 3.533321868089888,
 'openAcc': 3.4746484439407403,
 'n12': 3.2870250949999997,
 'n13': 3.1713307876666668,
 'n11': 2.1143802}
"""


def plot_fi_xgb(model, imp_type, model_nm):
  xgb_fi_dict = model.get_score(importance_type=imp_type)
  fi_df = pd.DataFrame(xgb_fi_dict.items(), columns=[
                       'feature_names', 'feature_importance'])
  fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

  plt.figure(figsize=(10, 8))
  sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
  plt.title(model_nm + 'FEATURE IMPORTANCE')
  plt.xlabel('FEATURE IMPORTANCE')
  plt.ylabel('FEATURE NAMES')


plot_fi_xgb(xgb_tuned, imp_type='gain', model_nm='XGBOOST ')
