from predictor import Predictor
from stock_parser import shift_df, load_train_test_data

import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

stock_df = pd.read_csv('../data/NVDA.csv')
stock_df = stock_df[['Date', 'Close']]

# hyperparameters
seq_len = 7
train_proportion = 0.95
batch_size = 5
num_epochs = 100
learning_rate = 0.003

stock_df_np = shift_df(stock_df, seq_len).to_numpy()[:, 1:]

train_loader, test_loader = load_train_test_data(stock_df_np, train_proportion, batch_size, seq_len)

stock_predictor = Predictor(input_size=1, hidden_size=seq_len+1, num_layers=1)

criterion = nn.MSELoss()
optimizer = optim.Adam(stock_predictor.parameters(), lr=learning_rate)

stock_predictor.train(train_loader, criterion, optimizer, num_epochs)
all_test_predictions, all_y_true, all_test_loss = stock_predictor.test(test_loader, criterion)

# for testing
# x_axis_epoch_loss = np.arange(len(all_epoch_loss))
# x_axis_all_loss = np.arange(len(all_loss))

# plt.subplot(2, 1, 1)
# plt.title('epoch loss')
# plt.plot(x_axis_epoch_loss, all_epoch_loss)

# plt.subplot(2, 1, 2)
# plt.title('batch loss')
# plt.plot(x_axis_all_loss, all_loss)

# plt.show()

# prediction plots
x_axis_all_true_y = np.arange(len(all_y_true))
x_axis_all_test_predictions = np.arange(len(all_test_predictions))
x_axis_all_test_lost = np.arange(len(all_test_loss))

plt.subplot(2, 1, 1)
plt.title('test results')
plt.plot(x_axis_all_test_predictions, all_test_predictions, color='blue')
plt.plot(x_axis_all_true_y, all_y_true, color='red')

plt.subplot(2, 1, 2)
plt.title('test loss')
plt.plot(x_axis_all_test_lost, all_test_loss)

plt.show()