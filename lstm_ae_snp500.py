from torch import nn, optim
import torch
import numpy as np
from torch.utils.data import DataLoader


def set_seed(seed):
    """Set random seed of the program"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
set_seed(0)

class LSTM_ae_snp500(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM_ae_snp500, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size, bias=False)
        # self.pred = nn.Linear..?


    def forward(self, x):
        output, (_, _) = self.encoder.forward(x)  # z is the last hidden state of the encoder.
        z = output[:, -1].repeat(1, output.shape[1]).view(output.shape)
        z2, (_, _) = self.decoder.forward(z)  # z2 is the last hidden state of the decoder.
        out1 = self.linear(z2)
        # out2 = prediction..
        return out1 #, out2


# trainset = ...
# validationset = ...
# testset = ...
trainset = None
validationset = None
testset = None


def train_AE(lr, batch_size, epochs, hidden_size, clip = None):
    trainLoader = DataLoader(trainset, batch_size=batch_size, shuffle=True)



















































