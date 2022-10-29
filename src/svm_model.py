from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from abstract_bc import MlModel
from sklearn import svm

class SVM_model(MlModel):
    grid_parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    def __init__(self):
        super().__init__()
        self.name = "SVM"
        self.base_model = svm.SVC()
        #self.model = GridSearchCV(self.base_model, self.grid_parameters)
        self.model = LogisticRegression(max_iter=1e6)