import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from syn_dat_gen import generate_synth_data


def set_seed(seed):
    """Set random seed of the program"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
set_seed(0)


class LSTM_AE(nn.Module):
    def __init__(self, input_size, hidden_size, gradient_clipping=False):
        super(LSTM_AE, self).__init__()
        # Regular Parameters
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Torch parameters
        self.encoder = nn.LSTM(input_size, hidden_size)
        self.decoder = nn.LSTM(hidden_size, hidden_size)
        # self.linear = nn.Linear(hidden_size, input_size, bias=False)            # Todo: Do we need this?


    def forward(self, x):
        z, (_, _) = self.encoder.forward(x)                 # z is the last hidden state of the encoder.
        z2, (_, _) = self.decoder.forward(z)                # z2 is the last hidden state of the decoder.
        # out = self.linear(z2)  # Todo: Do we need this?
        # out = F.tanh(out)
        # return z2
        return z2                   # Todo: Temporary, because we want the output, not the last hidden state.
                                    #  (Can also use the linear layer..).


trainset, validationset, testset = generate_synth_data(10000, 50)       # Generate synthetic data.

# validationloader = DataLoader(validationset, batch_size=batch_size, shuffle=False)

def train_AE(lr: float, batch_size: int, epochs: int):
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    model = LSTM_AE(1, 64)
    opt = optim.Adam(model.parameters(), lr)
    critireon = torch.nn.MSELoss()
    for epoch in range(epochs):
        total_loss = 0.0

        for i, data in enumerate(trainloader):
            opt.zero_grad()
            output = model(data)
            loss = critireon(data, output)
            loss.backward()
            opt.step()

        # Todo: How do we learn from validation data?

    # save the model:
    # PATH = './lstm_ae_synth.pth'
    # torch.save(model.state_dict(), PATH)  # SAVE THE  WEIGHTS OF THE MODEL (not all of it, can if want, but larger..).

    return model


# Todo: Testing = Taking the RMSE?
def test_model(model, batch_size):
    """Test the model on the test-data"""
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    model.eval()  # Change flag in parent model from true to false (train-flag).
    total_loss = 0
    with torch.no_grad():  # Everything below - will not calculate the gradients.
        # Iterate over the test data
        for data in testloader:
            # Apply model (forward pass).
            outputs = model(data)

            # Get predicted label
            total_loss += RMSE(data, outputs)                    # MSELoss of the output and data



model = train_AE(1e-3, 500, 1000)

















