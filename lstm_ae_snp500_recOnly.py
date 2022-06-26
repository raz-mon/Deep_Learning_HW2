from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def set_seed(seed):
    """Set random seed of the program"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)


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

# Get data
data = pd.read_csv('snp500_data/SP 500 Stock Prices 2014-2017.csv')
data = data[['symbol', 'high']]
names = data['symbol'].unique()
tss = []                            # An array of all the time-series (per symbol).
bad = []                            # An array of all the bad time-series - length not 1007.
for name in names:
    ts = data[data['symbol'] == name]['high']
    if not len(ts.values) == 1007 or np.isnan(ts).sum() != 0:
        bad += [ts.values]
        continue
    tss += [ts.values]
tss = np.array(tss)
orig_tss = tss.copy()



# Normalize the data, with the min-max normalization.
def min_max_norm(val, min, max):
    return (val-min) / (max - min)

"""class NormMinMax:
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def normalize_ts(self, ts):
        for ind in range(len(ts)):
            ts[ind] = min_max_norm(ts[ind], min, max)

    def unnormalize_ts(self):
        for ind in range(len(ts)):
            ts[ind] = ts[ind] * (self.max - self.min) + self.min"""

for ts in tss:
    min = np.min(ts)
    max = np.max(ts)
    for ind in range(len(ts)):
        ts[ind] = min_max_norm(ts[ind], min, max)


def unnormalize_tss(tss, init_ind):
    """
    Un-normalize time-series - according to their original min, max.
    :param tss: The data. Contains the time-series we wish to un-normalize.
    :type tss: Array of size (num_seq, seq_len).
    :param range: The range of these time-series in the original data-set (train: (0, int(len(orig_tss) * 0.6), validation: (int(len(orig_tss) * 0.6), int(len(orig_tss) * 0.8), test: You get it..
    :type range: Tuple of size 2: (begin_ind, end_ind).
    :return: Un-normalized data.
    :rtype: Array of the same size as the original one given (tss).
    """
    ret = []
    for ind in range(len(tss)):
        ts = tss[ind, :]
        min = np.min(orig_tss[init_ind + ind, :])
        max = np.max(orig_tss[init_ind + ind, :])
        for ind2 in range(len(ts)):
            ts[ind2] = ts[ind2] * (max - min) + min
        ret += [ts]
    return ret


def unnormalize_ts(ts, ind):
    ret = np.zeros(len(ts))
    min = np.min(orig_tss[ind, :])
    max = np.max(orig_tss[ind, :])
    for ind in range(len(ts)):
        ret[ind] = ts[ind] * (max - min) + min
    return ret


# Partition data to train, validation and test sets.
train_limit = int(len(tss) * 0.6)
validation_limit = int(len(tss) * 0.8)
train = tss[:train_limit, :]
validation = tss[train_limit:validation_limit, :]
test = tss[validation_limit:, :]

trainset = torch.tensor(train, dtype=torch.float32).view(len(train), len(train[0]), 1)       # Tensor of shape: (batch_size, seq_len, input_len) = (int(477*0.8), 1007, 1)
validationset = torch.tensor(validation, dtype=torch.float32).view(len(validation), len(validation[0]), 1)       # Tensor of shape: (batch_size, seq_len, input_len) = (int(477*0.2), 1007, 1)
testset = torch.tensor(test, dtype=torch.float32).view(len(test), len(test[0]), 1)       # Tensor of shape (approximately here): (batch_size, seq_len, input_len) = (int(477*0.2), 1007, 1)


def train_AE(lr, batch_size, epochs, hidden_size, clip=None, optimizer=None):
    trainLoader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    input_size = 1
    model = LSTM_ae_snp500(input_size, hidden_size)
    if optimizer is None:
        opt = optim.Adam(model.parameters(), lr)
    else:
        opt = optimizer
    critereon1 = nn.MSELoss()
    total_loss = 0.0

    for epoch in range(epochs):
        total_loss = 0.0

        for i, data in enumerate(trainLoader):
            opt.zero_grad()
            output = model(data)
            loss = critereon1(data, output)
            total_loss += loss.item()
            loss.backward()
            if clip is not None:
                clip_grad_norm_(model.parameters(), max_norm=clip)
            opt.step()

        print(f'epoch {epoch}, loss: {total_loss}')

    return model, total_loss


# Perform grid-search for the best hyper-parameters on the validation-set
def grid_search():
    counter = 0
    best_loss = float('inf')
    describe_model = None
    for hidden_state_size in [300]:
        for lr in [2e-3]:
            for batch_size in [5]:
                for grad_clipping in [2]:
                    print(f'\n\n\nModel num: {counter}, h_s_size: {hidden_state_size}, lr: {lr}, b_size: {batch_size}, g_clip: {grad_clipping}')
                    epochs = 60
                    model, loss = train_AE(lr, batch_size, epochs, hidden_state_size, grad_clipping)

                    if loss < best_loss:
                        best_loss = loss
                        describe_model = (counter, hidden_state_size, lr, batch_size, grad_clipping, loss)
                    validation_lost = test_validation(model)

                    # save the model:
                    file_name = f'ae_snp500_recOnly_{"Adam"}_lr={lr}_hidden_size={hidden_state_size}_gradient_clipping={grad_clipping}_batch_size{batch_size}' \
                                f'_epoch{epochs}_validation_loss_{validation_lost}'
                    path = os.path.join("saved_models", "snp500")
                    # create_folders(path)
                    torch.save(model, os.path.join(path, file_name + '.pt'))

    # print("best model {} params:\nhidden state: {}\nlearning state: {}\nbatch size: {}\ngrad clipping: {}\nloss: {}".format(*describe_model))


def test_validation(model, batch_size=None):
    # validationloader = DataLoader(validationset, batch_size=validationset.size()[0], shuffle=False)
    loss = torch.nn.MSELoss()
    model.eval()
    with torch.no_grad():
        output = model(validationset)
        curr_loss = loss(validationset, output)  # print("Accuracy: {:.4f}".format(acc))
    # print(f"validation loss = {curr_loss.item()}")
    model.train()
    return curr_loss.item()


def test_model(model):
    # testloader = DataLoader(testset, batch_size=testset.size()[0], shuffle=False)
    loss = torch.nn.MSELoss()
    model.eval()  # Change flag in parent model from true to false (train-flag).
    total_loss = 0
    with torch.no_grad():  # Everything below - will not calculate the gradients.
        outputs = model(testset)
        total_loss += loss(testset, outputs)  # MSELoss of the output and data
    model.train()
    return total_loss.item()


def test_train(model):
    loss = torch.nn.MSELoss()
    model.eval()  # Change flag in parent model from true to false (train-flag).
    total_loss = 0
    with torch.no_grad():  # Everything below - will not calculate the gradients.
        outputs = model(trainset)
        total_loss += loss(trainset, outputs)  # MSELoss of the output and data
    model.train()
    return total_loss.item()


def plot_google_amazon_high_stocks():
    stocks = pd.read_csv('snp500_data/SP 500 Stock Prices 2014-2017.csv')

    google_stocks = stocks[stocks['symbol'] == 'GOOGL'][['high', 'date']]
    google_stocks.index = google_stocks['date']
    google_stocks.index = pd.to_datetime(google_stocks.index)
    google_stocks['high'].plot(label='GOOGL')

    amazon_stocks = stocks[stocks['symbol'] == 'AMZN'][['high', 'date']]
    amazon_stocks.index = amazon_stocks['date']
    amazon_stocks.index = pd.to_datetime(amazon_stocks.index)
    amazon_stocks['high'].plot(label='AMZN')

    plt.title('Daily max value')
    plt.ylabel('value')
    plt.legend()
    plt.savefig('figures/Part3/GOOGL_AMZN_daily_max.png')
    plt.show()


def check_some_ts(model):
    xs = np.arange(0, 1007, 1)
    for ind in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]:
        ys = testset[ind, :, :]
        model.eval()
        ys_ae = model(ys).view(1007).detach().numpy()
        # ys_ae = unnormalize_ts(ys_ae, ind + int(len(orig_tss) * 0.8))
        model.train()
        ys = ys.view(1007).detach().numpy()
        plt.plot(xs, ys, label=f'orig')
        plt.plot(xs, ys_ae, label=f'rec')
        plt.title(f'Original and reconstructed signals - ind={ind}')
        plt.legend()
        plt.show()

# Set random seed to 0.
set_seed(0)

# Plot GOOGL and AMZN stocks from the data-set
"""plot_google_amazon_high_stocks()"""

# Perform a grid-search for the best hyper-parameters
"""grid_search()"""

# Load a model, and plot some of the reconstructions with the original signals to check how your model is doing
"""
model = torch.load("saved_models/snp500/ae_snp500_Adam_lr=0.002_hidden_size=100_gradient_clipping=0.9_batch_size50_epoch40_validation_loss_0.06462101638317108.pt")
check_some_ts(model)
# print(test_train(model))
"""


"""def plot_orig_and_reconstructed(model, ind):
    ts = orig_tss[ind, :]
    ts = np.array(ts)
    output = model(ts)
    ys = unnormalize_ts(output, ind)

    xs = np.arange(0, len(ts), 1)
    ys = ys.view(50).detach().numpy()
    plt.plot(xs, ys, label=f'orig')
    plt.plot(xs, ys_ae, label=f'rec')
    plt.title(f'Original and reconstructed signals - ind={ind}')"""
















