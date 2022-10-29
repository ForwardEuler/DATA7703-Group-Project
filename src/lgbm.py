from sklearn.model_selection import GridSearchCV
from abstract_bc import MlModel
from f_ndarray import msave, mload
from abstract_bc import printf
import numpy as np

from lightgbm import LGBMClassifier

class lgbm(MlModel):
    def __init__(self):
        super().__init__()
        self.name = "xgboost"
        self.model = LGBMClassifier()


if __name__ == '__main__':
    data = mload('train.npz')
    my_x = data.drop(['isDefault'], axis=1).to_numpy()
    my_y = data['isDefault'].to_numpy(dtype=np.int8)
    d_test = mload('test.npz')
    test_x = d_test.drop(['isDefault'], axis=1).to_numpy()
    test_y = d_test['isDefault'].to_numpy(dtype=np.int8)
    # my_x = mload('x_train_fe_v3.npz')
    # my_y = mload('y_train_fe_v3.npz')
    # test_x = mload('x_test_fe_v3.npz')
    # test_y = mload('y_test_fe_v3.npz')

    m = lgbm()
    m.train(my_x, my_y)
    m.evaluate(test_y, m.predict(test_x))
    p_y = m.predict(my_x)
    a = (p_y == my_y).mean()
    i = np.array(np.where(p_y == my_y))
    i = i.reshape(-1,)
    new_train = data.iloc[i]
    msave('new_train', new_train)
    printf("Train auc = %f\nTrain f1 = %f\n", m.train_auc, m.train_f1)
    printf("Test auc = %f\nTest f1 = %f\n", m.auc, m.f1)