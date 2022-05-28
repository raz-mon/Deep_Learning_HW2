import torch
from torch import nn, optim

torch.manual_seed(0)


class LSTM_AE(nn.Module):
    def __init__(self, input_size, hidden_size, num_epochs, optimizer, lr, gradient_clipping, batch_size):
        super(LSTM_AE, self).__init__()
        # Regular Parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.batch_size = batch_size

        # Torch parameters
        self.encoder = nn.LSTM(input_size, hidden_size)
        self.decoder = nn.LSTM(input_size, hidden_size)
        opt_dict = {"adam": optim.Adam, "sgd": optim.SGD, "mse": optim.RMSprop}
        self.optimizer = opt_dict[optimizer](self.parameters(), lr)

    def forward(self, x):
        output, _ = self.encoder.forward(x)
        output = output[:, -1]

        return output
