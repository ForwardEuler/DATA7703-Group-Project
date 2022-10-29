import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import f1_score

from f_ndarray import mload, msave
from lib_bc import *
from abstract_bc import BinaryClassification


class MLP(nn.Module, BinaryClassification):
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
        self.model = self.layers

    def forward(self, x):
        return self.layers(x)

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
        self.model = MLP()
        # copy to gpu
        self.model = self.model.to(device)

        # Define the loss function and optimizer
        loss_function = nn.BCELoss()
        # loss_function = loss_function.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        # optimizer = torch.optim.SGD(mlp.parameters(), lr=1e-3, momentum=0.9)
        # set max epoch
        epochs = 8
        for i in range(epochs):
            for x_train, y_train in tqdm(train_loader, unit="batch"):
                # copy data to GPU
                x_train = x_train.to(device)
                y_train = y_train.to(device)
                # FP
                output = self.model(x_train)
                # calculate loss
                loss = loss_function(output, y_train.reshape(-1, 1))

                optimizer.zero_grad()
                # Calculate the gradient
                loss.backward()
                # BP
                optimizer.step()
            predicted = self.model(torch.tensor(test_x, dtype=torch.float32).to(device))
            predicted = predicted.cpu()
            p_np = predicted.reshape(-1).detach().numpy()
            p_y = p_np.round()
            f1 = f1_score(test_y, p_y)
            printf("f1 = %f\n", f1)
        print('Training process has finished.')


if __name__ == '__main__':
    # check gpu available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    print(torch.cuda.get_device_name(0))

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
    mlp = MLP()
    mlp.fit(train_x, train_y)

    # # convert to tensor and set train batch size
    # tensor_x = torch.Tensor(train_x)
    # tensor_y = torch.Tensor(train_y)
    # dataset = TensorDataset(tensor_x, tensor_y)
    # train_loader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=True)
    #
    # # Initialize the MLP
    # mlp = MLP()
    # # copy to gpu
    # mlp = mlp.to(device)
    #
    # # Define the loss function and optimizer
    # loss_function = nn.BCELoss()
    # # loss_function = loss_function.to(device)
    # optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
    # # optimizer = torch.optim.SGD(mlp.parameters(), lr=1e-3, momentum=0.9)
    #
    # # set max epoch
    # epochs = 8
    # for i in range(epochs):
    #     for x_train, y_train in tqdm(train_loader, unit="batch"):
    #         # copy data to GPU
    #         x_train = x_train.to(device)
    #         y_train = y_train.to(device)
    #         # FP
    #         output = mlp(x_train)
    #         # calculate loss
    #         loss = loss_function(output, y_train.reshape(-1, 1))
    #
    #         optimizer.zero_grad()
    #         # Calculate the gradient
    #         loss.backward()
    #         # BP
    #         optimizer.step()
    #     predicted = mlp(torch.tensor(test_x, dtype=torch.float32).to(device))
    #     predicted = predicted.cpu()
    #     p_np = predicted.reshape(-1).detach().numpy()
    #     p_y = p_np.round()
    #     f1 = f1_score(test_y, p_y)
    #     printf("f1 = %f\n", f1)
    # print('Training process has finished.')
    #
    # # accuracy
    # predicted = mlp(torch.tensor(train_x, dtype=torch.float32).to(device))
    # predicted = predicted.cpu()
    # pred_np_x = predicted.reshape(-1).detach().numpy()
    # cutoff = optimal_cutoff(train_y, pred_np_x)
    # pred_y = score_map(pred_np_x, cutoff)
    # train_f1 = f1_score(train_y, pred_y)
    # train_auc = roc_auc_score(train_y, pred_y)
    # predicted = mlp(torch.tensor(test_x, dtype=torch.float32).to(device))
    # predicted = predicted.cpu()
    # pred_np = predicted.reshape(-1).detach().numpy()
    # cutoff = f1_cutoff(test_y, pred_np)
    # pred_y = score_map(pred_np, cutoff)
    # test_f1 = f1_score(test_y, pred_y)
    # test_auc = roc_auc_score(test_y, pred_np)
    # acc = (test_y == pred_y).mean()
    # plot_roc(test_y, pred_np, 'MLP')
    # cf = confusion_matrix(test_y, pred_y)
    # labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    # categories = ['Non-Default', 'Default']
    # make_confusion_matrix(cf,
    #                       group_names=labels,
    #                       categories=categories,
    #                       figsize=(12, 12),
    #                       cmap='Blues',
    #                       title="Confusion Matrix of")
