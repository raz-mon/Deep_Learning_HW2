import os
import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm
import numpy as np
from syn_dat_gen import generate_synth_data
import matplotlib.pyplot as plt
from pathlib import Path
import torchvision.transforms as transforms

ROW_BY_ROW_INPUT = 28
PIXEL_INPUT = 1


def set_seed(seed):
    """Set random seed of the program"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)


def create_folders(path):
    Path(path).mkdir(parents=True, exist_ok=True)


class LSTM_AE_MNIST(nn.Module):
    def __init__(self, input_size, hidden_size, is_reconstruct):
        super(LSTM_AE_MNIST, self).__init__()
        # Regular Parameters
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Torch parameters
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        if is_reconstruct:
            self.linear = nn.Linear(hidden_size, input_size, bias=False)  # Todo: Do we need this?
        else:
            self.linear = nn.Linear(hidden_size, 10, bias=False)

    def forward(self, x):
        output, (_, _) = self.encoder.forward(x)  # z is the last hidden state of the encoder.
        z = output[:, -1].repeat(1, output.shape[1]).view(output.shape)
        z2, (_, _) = self.decoder.forward(z)  # z2 is the last hidden state of the decoder.
        out = self.linear(z2)  # Todo: Do we need this?
        return out


def train_AE(input, lr: float, batch_size: int, epochs: int, hidden_size, clip: bool = None):
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    model = LSTM_AE_MNIST(input,
                    hidden_size, False)  # Choosing hidden_state_size to be smaller than the sequence_size, so we won't be learning the id function.
    opt = optim.Adam(model.parameters(), lr)
    critireon = torch.nn.MSELoss()
    total_loss = 0.0
    best_loss = float('inf')
    best_epoch = 0
    for epoch in range(epochs):
        total_loss = 0.0

        for i, data in enumerate(trainloader):
            opt.zero_grad()
            output = model(data)
            loss = critireon(data, output)
            total_loss += loss.item()
            loss.backward()
            if clip is not None:
                clip_grad_norm(model.parameters(), max_norm=clip)
            opt.step()

        print(f'epoch {epoch}, loss: {total_loss}')
        # Todo: How do we learn from validation data? By performing the grid-search below?
        if best_loss > total_loss:
            best_loss = total_loss
            best_epoch = epoch

    # save the model:
    file_name = f'ae_mnist_{"Adam"}_lr={lr}_hidden_size={hidden_size}_gradient_clipping={clip}_batch_size{batch_size}' \
                f'_epoch{epochs}_best_epoch{best_epoch}_best_loss{best_loss}'

    path = os.path.join("saved_models", "mnist_task")
    # create_folders(path)
    torch.save(model, os.path.join(path, file_name + '.pt'))

    return model, total_loss


# Perform grid-search for the best hyper-parameters on the validation-set
def grid_search():
    counter = 0
    best_loss = float('inf')
    describe_model = None
    for hidden_state_size in [30, 50, 100, 150]:
        for lr in [1e-2, 1e-3, 1e-4]:
            for batch_size in [32, 64]:
                for grad_clipping in [None, 0.9]:
                    print("Model num: ", counter)
                    counter += 1
                    _, loss = train_AE(lr, batch_size, 600, hidden_state_size, grad_clipping)
                    if loss < best_loss:
                        best_loss = loss
                        describe_model = (counter, hidden_state_size, lr, batch_size, grad_clipping, loss)
    print("best model {} params:\nhidden state: {}\nlearning state: {}\nbatch size: {}\ngrad clipping: {}\nloss: {}".format(*describe_model))


# Todo: Testing = Taking the RMSE?
def test_model(model):
    # testloader = DataLoader(testset, batch_size=testset.size()[0], shuffle=False)
    loss = torch.nn.MSELoss()
    # Test the model on the test-data
    model.eval()  # Change flag in parent model from true to false (train-flag).
    total_loss = 0
    #with torch.no_grad():  # Everything below - will not calculate the gradients.
        # for data in testset:
        # Apply model (forward pass).
        #outputs = model(testset)

        #total_loss += loss(testset, outputs)  # MSELoss of the output and data


set_seed(0)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((.5, .5, .5), (.5, .5, .5))])
trainset = torchvision.datasets.MNIST(root='../data/', train=True, download=True,
                                        transform=transform)

# grid_search()
