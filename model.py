import numpy as np
import torch
from torch import nn
import math


device = torch.device('cuda:1')


class ClassificationNet(nn.Module):
    def __init__(self, dimension):
        super(ClassificationNet, self).__init__()

        self.fc_1 = nn.Linear(dimension, 1024)
        self.fc_2 = nn.Linear(1024, 1024)
        self.fc_3 = nn.Linear(1024, 256)
        self.fc_4 = nn.Linear(256, 32)
        self.fc_5 = nn.Linear(32, 6)

    def forward(self, x):
        x = self.fc_1(x)
        x = torch.relu(x)
        x = self.fc_2(x)
        x = torch.relu(x)
        x = self.fc_3(x)
        x = torch.relu(x)
        x = self.fc_4(x)
        x = torch.relu(x)
        x = self.fc_5(x)
        x = nn.functional.softmax(x, dim=-1)

        return x


def train(
    model,
    train_dataloader,
    test_dataloader,
    model_path='/home/hanyuji/projects/PJ4/best_model',
    epochs=200,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    best_loss = math.inf
    model.train()

    for epoch in range(epochs):
        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

        # 保存当前最好的模型
        if epoch % 10 == 0:
            cur_loss = validation(model, test_dataloader, criterion)
            if cur_loss < best_loss:
                best_loss = cur_loss

                # 保存模型参数
                weights_dict = {}
                for k, v in model.state_dict().items():
                    weights_dict[k] = v
                torch.save(weights_dict, f'{model_path}/model_best.pkl')

            print(f'epoch: {epoch}, train_loss: {float(loss)}, val_loss: {best_loss}')


def validation(model, test_dataloader, criterion):
    total_lost = []
    model.eval()
    with torch.no_grad():
        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total_lost.append(loss.cpu().numpy())

    return np.array(total_lost).mean()
