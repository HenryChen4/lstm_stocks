import torch
import torch.nn as nn

# takes in 3 timesteps with each input being 1 dimension
rnn = nn.LSTM(input_size=1, hidden_size=3, num_layers=1, batch_first=True)

# dims: (num layers, batch size, num hidden units)
h0 = torch.randn(1, 1, 3)
c0 = torch.randn(1, 1, 3)

# dims: (batch size, sequence length, feature dim)
x = torch.randn(1, 2, 1)

out, _ = rnn(x, (h0, c0))
print(out)
print(out[:, -1, :])
fc = nn.Linear(3, 1)
print(fc(out[:, -1, :]))