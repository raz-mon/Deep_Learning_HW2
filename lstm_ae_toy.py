import os
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import numpy as np
import syn_dat_gen
from syn_dat_gen import generate_synth_data
import matplotlib.pyplot as plt
from pathlib import Path


def set_seed(seed):
    """Set random seed of the program"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)


def create_folders(path):
    Path(path).mkdir(parents=True, exist_ok=True)


class LSTM_AE(nn.Module):
    def __init__(self, input_size, hidden_size, gradient_clipping=False):
        super(LSTM_AE, self).__init__()
        # Regular Parameters
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Torch parameters
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size, bias=False)

    def forward(self, x):
        output, (_, _) = self.encoder.forward(x)  # z is the last hidden state of the encoder.
        z = output[:, -1].repeat(1, output.shape[1]).view(output.shape)
        z2, (_, _) = self.decoder.forward(z)  # z2 is the last hidden state of the decoder.
        out = self.linear(z2)
        out = torch.sigmoid(out)
        return out


def train_AE(lr: float, batch_size: int, epochs: int, hidden_size, clip):
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    model = LSTM_AE(1,
                    hidden_size)  # Choosing hidden_state_size to be smaller than the sequence_size, so we won't be learning the id function.
    opt = optim.Adam(model.parameters(), lr)
    criterion = torch.nn.MSELoss()
    total_loss = 0.0
    best_loss = float('inf')
    best_epoch = 0
    l = []
    for epoch in range(epochs):
        total_loss = 0.0

        for i, data in enumerate(trainloader):
            opt.zero_grad()
            output = model(data)
            loss = criterion(data, output)
            total_loss += loss.item()
            loss.backward()
            if clip is not None:
                clip_grad_norm_(model.parameters(), max_norm=clip)
            opt.step()
        l += [total_loss]

        print(f'epoch {epoch}, loss: {total_loss}')
        # Todo: How do we learn from validation data? By performing the grid-search below?
        if best_loss > total_loss:
            best_loss = total_loss
            best_epoch = epoch

    # save the model:
    file_name = f'ae_toy_{"Adam"}_lr={lr}_hidden_size={hidden_size}_gradient_clipping={clip}_batch_size{batch_size}' \
                f'_epoch{epochs}_best_epoch{best_epoch}_best_loss{best_loss}'

    path = os.path.join("saved_models", "toy_task")
    # create_folders(path)
    torch.save(model, os.path.join(path, file_name + '.pt'))

    return model, l


def save_plot_loss_vs_epochs(loss, batch_size, epochs, hidden_state_size, grad_clipping):
    plt.figure()
    plt.plot(np.arange(0, len(loss), 1), loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('loss vs. epoch')
    plt.savefig(f'figures/Part1/loss_vs_epochs_toy_bsize_{batch_size}_epochs_{epochs}_hidden_state_size_'
                f'{hidden_state_size}_g_clip{grad_clipping}')


# Perform grid-search for the best hyper-parameters on the validation-set
def grid_search():
    counter = 0
    best_loss = float('inf')
    describe_model = None
    for hidden_state_size in [20, 30, 40]:
        for lr in [1e-3, 2e-3, 5e-3]:
            for batch_size in [4, 8, 15, 50]:
                for grad_clipping in [0.8, 1, 1.2, 5]:
                    epochs = 2500
                    print(f'\n\n\nModel num: {counter}, h_s_size: {hidden_state_size}, lr: {lr}, b_size: {batch_size}, g_clip: {grad_clipping},'
                          f' epochs: {epochs}')

                    # counter += 1
                    # if counter < 43:
                    #     continue
                    _, loss = train_AE(lr, batch_size, epochs, hidden_state_size, grad_clipping)
                    if loss[-1] < best_loss:
                        best_loss = loss[-1]
                        describe_model = (counter, hidden_state_size, lr, batch_size, grad_clipping, loss[-1])
                    save_plot_loss_vs_epochs(loss, batch_size, epochs, hidden_state_size, grad_clipping)
    print("best model {} params:\nhidden state: {}\nlearning state: {}\nbatch size: {}\ngrad clipping: {}\nloss: {}".format(*describe_model))


def test_validation(model, batch_size=None):
    # validationloader = DataLoader(validationset, batch_size=validationset.size()[0], shuffle=False)
    loss = torch.nn.MSELoss()
    model.eval()
    with torch.no_grad():
        output = model(validationset)
        curr_loss = loss(validationset, output)  # print("Accuracy: {:.4f}".format(acc))
    # print(f"validation loss = {curr_loss.item()}")
    model.train()
    return curr_loss


def test_model(model):
    # testloader = DataLoader(testset, batch_size=testset.size()[0], shuffle=False)
    loss = torch.nn.MSELoss()
    model.eval()  # Change flag in parent model from true to false (train-flag).
    total_loss = 0
    with torch.no_grad():  # Everything below - will not calculate the gradients.
        outputs = model(testset)
        total_loss += loss(testset, outputs)  # MSELoss of the output and data
    model.train()
    return total_loss


def check_some_ts(model):
    xs = np.arange(0, 50, 1)
    for ind in [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]:
        ys = trainset[ind, :, :]
        model.eval()
        ys_ae = model(ys).view(50).detach().numpy()
        model.train()
        ys = ys.view(50).detach().numpy()
        plt.plot(xs, ys, label=f'orig')
        plt.plot(xs, ys_ae, label=f'rec')
        plt.xlabel('time')
        plt.ylabel('value')
        plt.title(f'Original and reconstructed signals - ind={ind}')
        plt.legend()
        plt.ylim((-0.25, 1.25))
        plt.show()


def print_reconstruct(model):
    for i in [1, 5, 10, 100, 200, 350, 450, 550, 563]:
        xs = np.arange(0, 50, 1)
        ys1 = testset[i, :, :]
        model.eval()
        ys2 = model(ys1).view(50).detach().numpy()
        plt.plot(xs, ys1.view(50).detach().numpy(), label='orig')
        plt.plot(xs, ys2, label='rec')
        plt.legend()
        plt.title('original and reconstructed signal')
        plt.show()


# Set random seed and generate data.
set_seed(0)
trainset, validationset, testset = generate_synth_data(10000, 50)  # Generate synthetic data.

# perfom a grid search for best hyper-parameters
"""grid_search()"""

# Load a model, and check some of the time-series in the data set on it.
"""model = torch.load(
    "saved_models/toy_task/ae_toy_Adam_lr=0.003_hidden_size=46_gradient_clipping=10_batch_size5_epoch400_best_epoch246_best_loss1.7050432766554877.pt")
check_some_ts(model)
# print_reconstruct()

# print(test_validation(model))
# print(test_model(model))
"""

# Generate plots of 3 examples of the data-set
"""syn_dat_gen.generate_and_plot()"""

# Train a model and test it
"""
model = train_AE(1e-3, 30, 20)
test_model(model)
"""

