import numpy as np
import torch
from sklearn.model_selection import train_test_split


class DataSet(torch.utils.data.Dataset):
    def __init__(self, data_inputs, data_targets):
        self.inputs = torch.FloatTensor(data_inputs)
        self.label = torch.LongTensor(data_targets)

    def __getitem__(self, index):
        return self.inputs[index], self.label[index]

    def __len__(self):
        return len(self.inputs)


def get_dataloader(features, labels, batch_Size=200):
    labels = labels -1
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=1,
        shuffle=True,
    )
    train_dataset = DataSet(np.array(x_train), list(y_train))
    test_dataset = DataSet(np.array(x_test), list(y_test))

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_Size, shuffle=True, drop_last=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_Size, shuffle=True, drop_last=True
    )
    return train_dataloader, test_dataloader


def get_test_dataloader(features, labels, batch_Size=200):
    labels = labels -1
    test_dataset = DataSet(np.array(features), list(labels))

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_Size, shuffle=True, drop_last=True
    )
    return test_dataloader
