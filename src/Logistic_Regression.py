from sklearn.metrics import roc_auc_score, f1_score

from abstract_bc import BinaryClassification
from sklearn.linear_model import LogisticRegression
from f_ndarray import *

class Logistic_model(BinaryClassification):
    def __init__(self):
        super().__init__()
        self.name = "Logistic Regression"
        self.model = LogisticRegression(solver='lbfgs', n_jobs=14, tol=1e-9, C=0.9, max_iter=500000)

    def predict_proba(self, x: ndarray) -> ndarray:
        return self.model.predict_proba(x)[:, 1]

if __name__ == '__main__':
    # load data set
    data = mload('train.npz')
    train_x = data.drop(['isDefault'], axis=1).to_numpy()
    train_y = data['isDefault'].to_numpy(dtype=np.int8)
    d_test = mload('test.npz')
    test_x = d_test.drop(['isDefault'], axis=1).to_numpy()
    test_y = d_test['isDefault'].to_numpy(dtype=np.int8)

    # train and test model
    lr = Logistic_model()
    lr.fit(train_x, train_y)
    lr.evaluate(test_x, test_y)
    lr.plot_roc()
    lr.plot_confusion_matrix()
