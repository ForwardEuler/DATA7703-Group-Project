from sklearn.model_selection import train_test_split
from lib_bc import printf
#from dtree import dtree_model
from svm_model import SVM_model
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#
# m1 = random_forest()
# m1.train(X_train, y_train)
# print("auc=" + str(m.train_auc))
# y_predict = m.predict(X_test)
# printf("%s train auc = %f\n", m.name, m.train_auc)
# m.evaluate(y_test, y_predict)
# printf("%s test auc = %f\n", m.name, m.auc)

model1 = SVM_model()
models = [model1]

for m in models:
    m.train(X_train, y_train)
    y_predict = m.predict(X_test)
    printf("%s train auc = %f\n", m.name, m.train_auc)
    m.evaluate(y_test, y_predict)
    printf("%s test auc = %f\n", m.name, m.auc)
    printf("%s best params by grid search is:\n %s\n", m.name, m.model.best_params_)