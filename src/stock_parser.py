from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np
import torch
import copy

class Stock_Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

def shift_df(stock_df, seq_len):
    stock_df = copy.deepcopy(stock_df)
    for i in range(1, seq_len + 1):
        stock_df[f'Close(t-{i})'] = stock_df['Close'].shift(i)
    stock_df.dropna(inplace=True)
    return stock_df

def load_train_test_data(shifted_stock_df_np, train_prop, batch_size, seq_len):
    stock_df = np.flip(shifted_stock_df_np, axis=1)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    stock_df = scaler.fit_transform(stock_df)

    all_x = stock_df[:, :-1]
    all_y = stock_df[:, -1]

    data_len = len(all_x)
    x_train = all_x[:int(data_len * train_prop)]
    y_train = all_y[:int(data_len * train_prop)]
    x_test = all_x[int(data_len * train_prop):]
    y_test = all_x[int(data_len * train_prop):]

    x_train = x_train.reshape((-1, seq_len, 1))
    y_train = y_train.reshape((-1, 1))

    x_test = x_test.reshape((-1, seq_len, 1))
    y_test = y_test.reshape((-1, 1))

    x_train = torch.tensor(x_train).float()
    y_train = torch.tensor(y_train).float()
    x_test = torch.tensor(x_test).float()
    y_test = torch.tensor(y_test).float()

    training_set = Stock_Dataset(x_train, y_train)
    test_set = Stock_Dataset(x_test, y_test)

    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader