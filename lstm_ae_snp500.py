from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import util


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
        self.pred = nn.Linear(hidden_size, input_size, bias=False)

    def forward(self, x):
        output, (_, _) = self.encoder.forward(x)  # z is the last hidden state of the encoder.
        z = output[:, -1].repeat(1, output.shape[1]).view(output.shape)
        z2, (_, _) = self.decoder.forward(z)  # z2 is the last hidden state of the decoder.
        rec = self.linear(z2)
        pred = self.pred(z2[:, :, :])
        return rec, pred


# Get data
data = pd.read_csv('snp500_data/SP 500 Stock Prices 2014-2017.csv')
data = data[['symbol', 'high', 'date']]
names = data['symbol'].unique()
tss = []  # An array of all the time-series (per symbol).
bad = []  # An array of all the bad time-series - length not 1007.
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
    return (val - min) / (max - min)


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


def unnormalize_ts_min_max(ts, min, max):
    for i in range(len(ts)):
        ts[i] = ts[i] * (max - min) + min
    return ts


def train_AE(lr, batch_size, epochs, hidden_size, clip=None, optimizer=None):
    trainLoader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    input_size = 1
    model = LSTM_ae_snp500(input_size, hidden_size)
    if optimizer is None:
        opt = optim.Adam(model.parameters(), lr)
    else:
        opt = optimizer
    critereon1 = nn.MSELoss()
    total_loss = []
    rec_loss = []
    pred_loss = []

    for epoch in range(epochs):
        curr_tot_loss = 0.0
        curr_loss_rec = 0.0
        curr_loss_pred = 0.0
        for i, data in enumerate(trainLoader):
            correct_preds = data[:, 1:, :]

            opt.zero_grad()
            output_rec, output_pred = model(data)
            l_rec = critereon1(data, output_rec)
            l_pred = critereon1(correct_preds, output_pred[:, :-1, :])
            loss = l_rec + l_pred
            curr_loss_rec += l_rec.item()
            curr_loss_pred += l_pred.item()
            curr_tot_loss += l_rec.item() + l_pred.item()
            loss.backward()
            if clip is not None:
                clip_grad_norm_(model.parameters(), max_norm=clip)
            opt.step()

        total_loss += [curr_tot_loss]
        rec_loss += [curr_loss_rec]
        pred_loss += [curr_loss_pred]
        print(f'epoch {epoch}, loss: {total_loss[-1]}')

        if not (epoch + 1) % 300:
            savefigs(rec_loss, pred_loss, hidden_size, lr, batch_size, clip, epoch)

            # save the model:
            file_name = f'ae_snp500_pred_{"Adam"}_lr={lr}_hidden_size={hidden_size}_gradient_clipping={clip}_batch_size{batch_size}' \
                        f'_epoch{epoch}_validation_loss_{test_validation(model)}'
            path = os.path.join("saved_models", "snp500")
            # create_folders(path)
            torch.save(model, os.path.join(path, file_name + '.pt'))

    return model, total_loss, rec_loss, pred_loss


# Perform grid-search for the best hyper-parameters on the validation-set
def grid_search():
    best_loss = float('inf')
    describe_model = None
    for hidden_state_size in [120]:
        for lr in [1e-3]:
            for batch_size in [10]:
                for grad_clipping in [1]:
                    print(
                        f'\n\n\nh_s_size: {hidden_state_size}, lr: {lr}, b_size: {batch_size}, g_clip: {grad_clipping}')
                    epochs = 3000
                    model, loss, rec_loss, pred_loss = train_AE(lr, batch_size, epochs, hidden_state_size,
                                                                grad_clipping)
                    # savefigs(rec_loss, pred_loss, hidden_state_size, lr, batch_size, grad_clipping, epochs)

                    validation_loss = test_validation(model)

                    if validation_loss < best_loss:
                        best_loss = validation_loss
                        describe_model = (hidden_state_size, lr, batch_size, grad_clipping, epochs, validation_loss)
                    """
                    # save the model:
                    file_name = f'ae_snp500_pred_{"Adam"}_lr={lr}_hidden_size={hidden_state_size}_gradient_clipping={grad_clipping}_batch_size{batch_size}' \
                                f'_epoch{epochs}_validation_loss_{validation_loss}'
                    path = os.path.join("saved_models", "snp500")
                    # create_folders(path)
                    torch.save(model, os.path.join(path, file_name + '.pt'))
                    """

    print(
        "best model params:\nhidden state: {}\nlearning state: {}\nbatch size: {}\ngrad clipping: {}\nepochs: {}\nvalidation loss: {}".format(
            *describe_model))


def test_validation(model, batch_size=None):
    # validationloader = DataLoader(validationset, batch_size=validationset.size()[0], shuffle=False)
    loss = torch.nn.MSELoss()
    model.eval()
    with torch.no_grad():
        rec_output, pred_output = model(validationset)
        curr_loss = loss(validationset, rec_output) + loss(validationset[:, 1:, :], pred_output[:, :-1,
                                                                                    :])  # print("Accuracy: {:.4f}".format(acc))
    # print(f"validation loss = {curr_loss.item()}")
    model.train()
    return curr_loss.item()


def test_model(model):
    # testloader = DataLoader(testset, batch_size=testset.size()[0], shuffle=False)
    loss = torch.nn.MSELoss()
    model.eval()
    with torch.no_grad():
        rec_output, pred_output = model(validationset)
        curr_loss = loss(testset, rec_output) + loss(validationset[:, 1:, :],
                                                     pred_output[:, :-1, :])  # print("Accuracy: {:.4f}".format(acc))
    model.train()
    return curr_loss.item()


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
    data = pd.read_csv('snp500_data/SP 500 Stock Prices 2014-2017.csv')
    # xs = np.arange(0, 1007, 1)
    xs = data[data['symbol'] == 'GOOGL']['date']
    for ind in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]:
        ys = testset[ind, :, :]
        model.eval()
        ys_rec, ys_pred = model(ys.view(1, len(ys), 1))
        ys_rec = ys_rec.view(1007).detach().numpy()
        ys_pred = ys_pred.view(1007).detach().numpy()
        # ys_ae = unnormalize_ts(ys_ae, ind + int(len(orig_tss) * 0.8))
        model.train()
        ys = ys.view(1007).detach().numpy()
        ys = unnormalize_ts(ys, ind)
        ys_rec = unnormalize_ts(ys_rec, ind)
        ys_pred = unnormalize_ts(ys_pred, ind)
        df = pd.DataFrame(data={'date': xs, 'ys_orig': ys, 'ys_rec': ys_rec})
        df2 = pd.DataFrame(data={'date': xs, 'ys_pred': ys_pred})
        df.index = df['date']
        df2.index = df2['date']
        df['ys_orig'].plot(label='orig')
        df['ys_rec'].plot(label='rec')
        df2['ys_pred'].plot(label='pred')
        # plt.title(f'Original and reconstructed signals - ind={ind}')
        plt.title(f'Original, reconstructed and predicted signals - ind={ind}')
        plt.ylabel('value')
        plt.xticks(rotation=5)
        plt.legend()
        plt.show()


"""def check_some_ts_half(model):
    # xs = np.arange(0, 500, 1)
    xs = data['date'][:500]
    for ind in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]:
        ys = testset[ind, :, :]
        model.eval()
        ys_rec, ys_pred = model(ys.view(1, len(ys), 1))
        ys_rec = ys_rec.view(500).detach().numpy()
        ys_pred = ys_pred.view(499).detach().numpy()
        # ys_ae = unnormalize_ts(ys_ae, ind + int(len(orig_tss) * 0.8))
        model.train()
        ys = ys.view(500).detach().numpy()
        plt.plot(xs, ys, label=f'orig')
        plt.plot(xs, ys_rec, label=f'rec')
        plt.plot(xs[:-1], ys_pred, label=f'pred')
        plt.title(f'Original, reconstructed and predicted signals - ind={ind}')
        plt.ylabel('value')
        plt.xticks(rotation=5)
        plt.legend()
        plt.show()"""

"""
def savefigs(rec_loss, pred_loss, hidden_state_size, lr, batch_size, grad_clipping, epochs):
    xs = [i for i in range(epochs)]
    plt.figure()
    plt.plot(xs, rec_loss, label='Reconstruction loss vs. epochs')
    plt.title('Reconstruction loss vs. epochs')
    plt.xlabel('date')
    plt.ylabel('value')
    plt.legend()
    plt.savefig(
        f'figures/Part3/rec_epochs_hidden_{hidden_state_size}_lr_{lr}_batchSize_{batch_size}_gradClip_{grad_clipping}_epochs_{epochs}.png')

    plt.figure()
    plt.plot(xs, pred_loss, label='Prediction loss vs. epochs')
    plt.title('Prediction loss vs. epochs')
    plt.xlabel('date')
    plt.ylabel('value')
    plt.legend()
    plt.savefig(
        f'figures/Part3/pred_epochs_hidden_{hidden_state_size}_lr_{lr}_batchSize_{batch_size}_gradClip_{grad_clipping}_epochs_{epochs}.png')

    # plt.show()
"""


def savefigs(rec_loss, pred_loss, hidden_state_size, lr, batch_size, grad_clipping, epochs):
    xs = [i for i in range(len(rec_loss))]
    axis = util.initiate_graph(1, 2)
    util.plot_multi_graph(axis, 0, 0, xs, [rec_loss], ['Reconstruction loss vs. epochs'],
                          "S&P500 - Reconstruction", "epoch",
                          "loss", 1)
    util.plot_multi_graph(axis, 1, 0, xs, [pred_loss], ['Prediction loss vs. epochs'],
                          "S&P500 - Prediction", "epoch",
                          "loss", 1)
    plt.savefig(
        f'figures/Part3/pred_epochs_hidden_{hidden_state_size}_lr_{lr}_batchSize_{batch_size}_gradClip_{grad_clipping}_epochs_{epochs}.png')
    # plt.show()


def multi_step_prediction(model):
    stocks = pd.read_csv('snp500_data/SP 500 Stock Prices 2014-2017.csv')
    google_stocks = stocks[stocks['symbol'] == 'GOOGL'][["date", "high"]]

    ts = google_stocks['high'].values
    dates = pd.to_datetime(google_stocks['date'])

    ts2 = ts.copy()
    mini = np.min(list(ts2))
    maxi = np.max(list(ts2))
    ts2 = min_max_norm(ts2, mini, maxi)
    past_ts = torch.tensor(ts2[:len(ts2) // 2], dtype=torch.float32).view(1, len(ts2) // 2, 1)
    # pred_ts = torch.tensor(ts[len(ts) // 2:], dtype=torch.float32).view(1, len(ts) // 2, 1)
    # for i in range(len(ts) // 2):
    for i in range(503):
        print(i)
        _, past_ts = model(past_ts)

    temp = list(past_ts.view(len(ts) // 2, 1).detach().numpy().T[0])
    temp = unnormalize_ts_min_max(temp, np.min(ts[len(ts) // 2:]), np.max(ts[len(ts) // 2:]))

    _, axis1 = plt.subplots(1, 1)
    axis1.plot(list(dates), list(ts), label="google")
    axis1.plot(list(dates), [0] * (len(ts) // 2 + 1) + temp, label="prediction")
    plt.xticks(rotation=30)
    plt.title("google max stock values, years 2014-2017")
    plt.legend()
    plt.show()

    print("done")


# plot_google_amazon_high_stocks()


# Get data

data = pd.read_csv('snp500_data/SP 500 Stock Prices 2014-2017.csv')
data = data[['symbol', 'high']]
names = data['symbol'].unique()
tss = []  # An array of all the time-series (per symbol).
bad = []  # An array of all the bad time-series - length not 1007.
for name in names:
    ts = data[data['symbol'] == name]['high']
    if not len(ts.values) == 1007 or np.isnan(ts).sum() != 0:
        bad += [ts.values]
        continue
    tss += [ts.values]
tss = np.array(tss)
orig_tss = tss.copy()

for ts in tss:
    min = np.min(ts)
    max = np.max(ts)
    for ind in range(len(ts)):
        ts[ind] = min_max_norm(ts[ind], min, max)

# Partition data to train, validation and test sets.
train_limit = int(len(tss) * 0.6)
validation_limit = int(len(tss) * 0.8)
train = tss[:train_limit, :]
validation = tss[train_limit:validation_limit, :]
test = tss[validation_limit:, :]
trainset = torch.tensor(train, dtype=torch.float32).view(len(train), len(train[0]),
                                                         1)  # Tensor of shape: (batch_size, seq_len, input_len) = (int(477*0.8), 1007, 1)
validationset = torch.tensor(validation, dtype=torch.float32).view(len(validation), len(validation[0]),
                                                                   1)  # Tensor of shape: (batch_size, seq_len, input_len) = (int(477*0.2), 1007, 1)
testset = torch.tensor(test, dtype=torch.float32).view(len(test), len(test[0]),
                                                       1)  # Tensor of shape (approximately here): (batch_size, seq_len, input_len) = (int(477*0.2), 1007, 1)

# grid_search()
# model = torch.load(
#     "saved_models/snp500/ae_snp500_pred_Adam_lr=0.002_hidden_size=300_gradient_clipping=3_batch_size10_epoch100_validation_loss_0.097300224006176.pt")
# multi_step_prediction(model)
# check_some_ts(model)


# grid_search()

# model = torch.load("saved_models/snp500/ae_snp500_pred_Adam_lr=0.001_hidden_size=120_gradient_clipping=1_batch_size10_epoch299_validation_loss_0.09807705879211426.pt")
# check_some_ts(model)
# multi_step_prediction(model)


model = torch.load(
    "saved_models/snp500/ae_snp500_pred_Adam_lr=0.001_hidden_size=120_gradient_clipping=1_batch_size10_epoch1199_validation_loss_0.02652263641357422.pt")
# check_some_ts(model)
multi_step_prediction(model)



"""def plot_orig_and_reconstructed(model, ind):

trainset = torch.tensor(train, dtype=torch.float32).view(len(train), len(train[0]),
                                                         1)  # Tensor of shape: (batch_size, seq_len, input_len) = (int(477*0.8), 1007, 1)
validationset = torch.tensor(validation, dtype=torch.float32).view(len(validation), len(validation[0]),
                                                                   1)  # Tensor of shape: (batch_size, seq_len, input_len) = (int(477*0.2), 1007, 1)
testset = torch.tensor(test, dtype=torch.float32).view(len(test), len(test[0]),
                                                       1)  # Tensor of shape (approximately here): (batch_size, seq_len, input_len) = (int(477*0.2), 1007, 1)

grid_search()
"""
# multi_step_prediction(model)
# check_some_ts(model)


# model = torch.load("saved_models/snp500/ae_snp500_pred_Adam_lr=0.002_hidden_size=300_gradient_clipping=3_batch_size10_epoch100_validation_loss_0.097300224006176.pt")
# check_some_ts(model)

# model = torch.load("saved_models/snp500/ae_snp500_pred_Adam_lr=0.002_hidden_size=300_gradient_clipping=3_batch_size4_epoch60_validation_loss_0.12832608819007874.pt")
# check_some_ts(model)


"""
def plot_orig_and_reconstructed(model, ind):
    ts = orig_tss[ind, :]
    ts = np.array(ts)
    output = model(ts)
    ys = unnormalize_ts(output, ind)

    xs = np.arange(0, len(ts), 1)
    ys = ys.view(50).detach().numpy()
    plt.plot(xs, ys, label=f'orig')
    plt.plot(xs, ys_ae, label=f'rec')
    plt.title(f'Original and reconstructed signals - ind={ind}')"""
