import torch
import torch.nn as nn

class Predictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        prediction = self.output_layer(lstm_out[:, -1, :])
        return prediction

    def train(self, train_loader, criterion, optimizer, epochs):
        all_loss = []
        all_epoch_loss = []

        for c in range(epochs):
            epoch_loss = 0.
            num_batches = len(train_loader)
            for i, train_tuple in enumerate(train_loader):
                x_train = train_tuple[0]
                y_true = train_tuple[1]
                
                predictions = self.forward(x_train)
                loss = criterion(predictions, y_true)

                all_loss.append(loss.item())
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            epoch_loss /= num_batches
            print(f"Epoch: {c}, Epoch Loss: {epoch_loss}")
            all_epoch_loss.append(epoch_loss)
        
        return all_epoch_loss, all_loss

    def test(self, test_loader, criterion):
        test_loss = []
        all_predictions = []
        all_y_true = []
        for i, test_tuple in enumerate(test_loader):
            x_test = test_tuple[0]
            y_test = test_tuple[1]
            
            predictions = self.forward(x_test)
            all_predictions.append(predictions)
            
            loss = criterion(predictions, y_test)
            test_loss.append(loss)
        
        return all_predictions, all_y_true, test_loss