import os
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import numpy as np
from syn_dat_gen import generate_synth_data
import matplotlib.pyplot as plt
from pathlib import Path
import util


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


def train_AE(lr: float, batch_size: int, epochs: int, hidden_size, clip = None):
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    # trainloader = DataLoader(trainset, shuffle=True)
    model = LSTM_AE(1,
                    hidden_size)  # Choosing hidden_state_size to be smaller than the sequence_size, so we won't be learning the id function.
    opt = optim.Adam(model.parameters(), lr)
    criterion = torch.nn.MSELoss()
    total_loss = 0.0
    best_loss = float('inf')
    best_epoch = 0
    loss = 0
    losses = []
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

        print(f'epoch {epoch}, loss: {total_loss}')
        # Todo: How do we learn from validation data? By performing the grid-search below?
        if best_loss > total_loss:
            best_loss = total_loss
            best_epoch = epoch
        losses.append(float(loss))
    # save the model:
    file_name = f'ae_toy_{"Adam"}_lr={lr}_hidden_size={hidden_size}_gradient_clipping={clip}_batch_size{batch_size}' \
                f'_epoch{epochs}_best_epoch{best_epoch}_best_loss{best_loss}'

    path = os.path.join("saved_models", "toy_task")
    # create_folders(path)
    torch.save(model, os.path.join(path, file_name + '.pt'))

    return losses, model, total_loss


# Perform grid-search for the best hyper-parameters on the validation-set
def grid_search():
    counter = 0
    best_loss = float('inf')
    describe_model = None
    epochs_x = [i for i in range(1, 431)]
    losses = []
    for hidden_state_size in [47]:
        for lr in [1e-3]:
            for batch_size in [6]:
                for grad_clipping in [1]:
                    epochs = 430
                    print(f'\n\n\nModel num: {counter}, h_s_size: {hidden_state_size}, lr: {lr}, b_size: {batch_size}, g_clip: {grad_clipping},'
                          f' epochs: {epochs}')
                    # counter += 1
                    # if counter < 43:
                    #     continue
                    curr_losses, _, loss = train_AE(lr, batch_size, epochs, hidden_state_size, grad_clipping)

                    losses.append(curr_losses)
                    plt.rc("font", size=12, family="Times New Roman")
                    plt.plot(epochs_x, curr_losses, label=f'h s: {hidden_state_size}, lr: {lr}, bt s: {batch_size}, clip: {grad_clipping}')
                    plt.xlabel("epoch")
                    plt.ylabel("loss")
                    plt.title("Toy - LSTM Autoencoder Model")
                    plt.legend()
                    plt.show()

                    if loss < best_loss:
                        best_loss = loss
                        describe_model = (counter, hidden_state_size, lr, batch_size, grad_clipping, loss)
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



set_seed(0)
trainset, validationset, testset = generate_synth_data(400, 50)  # Generate synthetic data.
# grid_search()


model = torch.load("saved_models/toy_task/ae_toy_Adam_lr=0.001_hidden_size=47_gradient_clipping=1_batch_size6_epoch430_best_epoch418_best_loss1.8480003140866756.pt")
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
        plt.show()
check_some_ts(model)

print(test_validation(model))
print(test_model(model))



# model = train_AE(1e-3, 30, 20)
# test_model(model)
#model = torch.load("saved_models/toy_task/ae_toy_Adam_lr=0.01_hidden_size=30__gradient_clipping=0.9_batch_size64_epoch=600.pt")
# print a ts and a reconstruction of it.
"""
xs = np.arange(0, 50, 1)
ys1 = testset[100, :, :]
model.eval()
ys2 = model(ys1).view(50).detach().numpy()
plt.plot(xs, ys1.view(50).detach().numpy(), label='orig')
plt.plot(xs, ys2, label='rec')
plt.legend()
plt.title('original and reconstructed signal')
plt.show()
"""


"""
AE = train_AE(1e-3, 500, 400)
test_model(AE)
ys1 = testset[0, :, :]
ys2 = AE(ys1).view(50).detach().numpy()
ys2 = AE(ys1).view(50).detach().numpy()
plt.plot(xs, ys1.view(50).detach().numpy(), label='orig')
plt.plot(xs, ys2, label='rec')
plt.legend()
plt.title('original and reconstructed signal')
plt.show()
"""
