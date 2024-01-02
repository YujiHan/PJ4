import numpy as np
import torch
from torch import nn
import math
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report


device = torch.device('cuda')


class ClassificationNet(nn.Module):
    def __init__(self, dimension):
        super(ClassificationNet, self).__init__()

        self.fc_1 = nn.Linear(dimension, 256)
        self.fc_2 = nn.Linear(256, 32)
        self.fc_3 = nn.Linear(32, 6)

    def forward(self, x):
        x = self.fc_1(x)
        x = torch.relu(x)
        x = self.fc_2(x)
        x = torch.relu(x)
        x = self.fc_3(x)

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
            cur_loss, accuracy, f1, f1_report = test_model(model, test_dataloader)
            if cur_loss < best_loss:
                best_loss = cur_loss

                # 保存模型参数
                weights_dict = {}
                for k, v in model.state_dict().items():
                    weights_dict[k] = v
                torch.save(weights_dict, f'{model_path}/model_best.pkl')

            print(f'epoch: {epoch}, train_loss: {float(loss)}, val_loss: {cur_loss}, accuracy: {accuracy}, f1: {f1}')


def test_model(model, test_dataloader):
    all_loss = []
    all_preds = []
    all_targets = []
    
    criterion = nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            all_loss.append(loss.cpu().numpy())
            
            _, predicted = torch.max(pred, 1)  # 获取最高概率的预测结果
            # 将预测和真实标签添加到列表中
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y.cpu().numpy())


    loss_average = np.array(all_loss).mean()
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted')  # 'weighted' 可以处理不平衡的类别
    f1_report = classification_report(all_targets, all_preds, zero_division=1)
    
    return loss_average, accuracy, f1, f1_report
