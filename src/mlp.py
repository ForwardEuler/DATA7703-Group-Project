import torch
from torch import nn, tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import f1_score
from f_ndarray import mload, msave
from lib_bc import *
from abstract_bc import BinaryClassification


class MLP_Struct(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(39, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


class MLP(BinaryClassification):
    def __init__(self):
        super().__init__()
        self.name = 'MLP'
        self.cutoff = 0.5

    def print(self):
        print(self.model)

    def fit(self, x: ndarray, y: ndarray) -> None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(device)
        print(torch.cuda.get_device_name(0))
        train_x, train_y = x, y
        tensor_x = torch.Tensor(train_x)
        tensor_y = torch.Tensor(train_y)
        dataset = TensorDataset(tensor_x, tensor_y)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=True)

        # Initialize the MLP
        mlp = MLP_Struct()
        # copy to gpu
        mlp = mlp.to(device)

        # Define the loss function and optimizer
        loss_function = nn.BCELoss()
        # loss_function = loss_function.to(device)
        optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
        # optimizer = torch.optim.SGD(mlp.parameters(), lr=1e-3, momentum=0.9)
        # set max epoch
        epochs = 15
        for i in range(epochs):
            for x_train, y_train in tqdm(train_loader, unit="batch"):
                # copy data to GPU
                x_train = x_train.to(device)
                y_train = y_train.to(device)
                # FP
                output = mlp(x_train)
                # calculate loss
                loss = loss_function(output, y_train.reshape(-1, 1))

                optimizer.zero_grad()
                # Calculate the gradient
                loss.backward()
                # BP
                optimizer.step()
            predicted = mlp(torch.tensor(test_x, dtype=torch.float32).to(device))
            predicted = predicted.cpu()
            p_np = predicted.reshape(-1).detach().numpy()
            auc = roc_auc_score(test_y, p_np)
            printf("auc = %f\n", auc)
        # save model
        mlp.cpu()
        self.model = mlp
        # save train score
        y_proba = mlp(tensor_x).reshape(-1).detach().numpy()
        self.train_auc = roc_auc_score(train_y, y_proba)
        self.cutoff = optimal_cutoff(train_y, y_proba)
        y_hat = score_map(y_proba, self.cutoff)
        self.train_f1 = f1_score(train_y, y_hat)
        print('Training process has finished.')

    def predict_proba(self, x: ndarray) -> ndarray:
        tensor_x = torch.tensor(x, dtype=torch.float32)
        predicted: tensor = self.model(tensor_x)
        predicted = predicted.reshape(-1).detach().numpy()
        return predicted

    def predict(self, x: ndarray) -> ndarray:
        y_proba = self.predict_proba(x)
        y_hat = score_map(y_proba, self.cutoff)
        return y_hat

    def evaluate(self, x_test: ndarray, y_test: ndarray) -> None:
        """
        evaluate model using test set
        """
        self._eval_y_true = y_test
        self._eval_y_proba = self.predict_proba(x_test)
        self.auc = roc_auc_score(y_test, self._eval_y_proba)
        self._eval_y_hat = self.predict(x_test)
        self.f1 = f1_score(y_test, self._eval_y_hat)


if __name__ == '__main__':
    # load data set
    torch.manual_seed(42)
    data = mload('train.npz')
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    train_x = data.drop(['isDefault'], axis=1).to_numpy()
    train_y = data['isDefault'].to_numpy(dtype=np.int8)
    d_test = mload('test.npz')
    d_test = d_test.sample(frac=1, random_state=42).reset_index(drop=True)
    test_x = d_test.drop(['isDefault'], axis=1).to_numpy()
    test_y = d_test['isDefault'].to_numpy(dtype=np.int8)

    # train and test model
    mlp_ = MLP()
    mlp_.fit(train_x, train_y)
    mlp_.evaluate(test_x, test_y)
    mlp_.plot_roc()
    mlp_.plot_confusion_matrix()
