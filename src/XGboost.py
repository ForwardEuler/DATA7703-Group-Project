from sklearn.model_selection import GridSearchCV
from abstract_bc import MlModel
from xgboost import XGBClassifier
from f_ndarray import msave, mload
from abstract_bc import printf
import numpy as np

class xgboost(MlModel):
    def __init__(self):
        super().__init__()
        self.name = "xgboost"
        self.model = XGBClassifier(tree_method='gpu_hist', gpu_id=0, max_depth=11)


if __name__ == '__main__':
    # data = mload('train_fe.npz')
    # my_x = data.drop(['isDefault'], axis=1).to_numpy()
    # my_y = data['isDefault'].to_numpy(dtype=np.int8)
    # d_test = mload('test_fe.npz')
    # test_x = d_test.drop(['isDefault'], axis=1).to_numpy()
    # test_y = d_test['isDefault'].to_numpy(dtype=np.int8)
    my_x = mload('x_train_fe_v3.npz')
    my_y = mload('y_train_fe_v3.npz')
    test_x = mload('x_test_fe_v3.npz')
    test_y = mload('y_test_fe_v3.npz')
    m = xgboost()
    m.train(my_x, my_y)
    m.evaluate(test_y, m.predict(test_x))
    printf("Train auc = %f\nTrain f1 = %f\n", m.train_auc, m.train_f1)
    printf("Test auc = %f\nTest f1 = %f\n", m.auc, m.f1)