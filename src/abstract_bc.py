from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import f1_score
from lib_bc import *
from sklearn.linear_model import LogisticRegression


class BinaryClassification:
    def __init__(self):
        self.name: str = "default"
        self.model = None
        self.train_auc: float = -0.0
        self.train_f1: float = -0.0
        self.auc: float = -0.0
        self.f1: float = -0.0
        self._eval_y_true = None
        self._eval_y_proba = None
        self._eval_y_hat = None

    def fit(self, x: ndarray, y: ndarray) -> None:
        """
        train model by x_train and y_train
        this method should also calculate auc and f1 for training set and store them into train_auc and train_f1
        """
        self.model.fit(x, y)
        y_predict: ndarray = self.model.predict_proba(x)
        self.train_auc = roc_auc_score(y, y_predict)
        y_hat = self.model.predict(x)
        self.train_f1 = f1_score(y, y_hat)

    def predict(self, x: ndarray) -> ndarray:
        return self.model.predict(x)

    def predict_proba(self, x: ndarray) -> ndarray:
        return self.model.predict_proba(x)

    def score(self, x: ndarray, y: ndarray) -> float:
        return self.model.score(x, y)

    def evaluate(self, x_test: ndarray, y_test: ndarray) -> None:
        """
        evaluate model using test set
        """
        self._eval_y_true = y_test
        self._eval_y_proba = self.model.predict_proba(x_test)
        self.auc = roc_auc_score(y_test, self._eval_y_proba)
        self._eval_y_hat = self.model.predict(x_test)
        self.f1 = f1_score(y_test, self._eval_y_hat)

    def plot_roc(self, name: str = None) -> None:
        """
        Can only be called after evaluate is called
        :param name: title of plot
        :return: None
        """
        if name is None:
            name = self.name
        plot_roc(self._eval_y_true, self._eval_y_proba, name=name)

    def plot_confusion_matrix(self, categories: list = None, name: str = None) -> None:
        """
        Can only be called after evaluate is called
        """
        if categories is None:
            categories = ['Negative', 'Positive']
        if name is None:
            name = self.name
        cf = confusion_matrix(self._eval_y_true, self._eval_y_hat)
        labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
        make_confusion_matrix(cf,
                              group_names=labels,
                              categories=categories,
                              figsize=(12, 12),
                              cmap='Blues',
                              title="Confusion Matrix of " + name)

    def print(self):
        raise NotImplementedError('In %s print is not implemented' % self.__class__)
