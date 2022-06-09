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


data = pd.read_csv('snp500_data/SP 500 Stock Prices 2014-2017.csv')
# trainset = ...
# validationset = ...
# testset = ...
trainset = None
validationset = None
testset = None


def train_AE(lr, batch_size, epochs, hidden_size, clip=None, optimizer=None):
    trainLoader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
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
    # for hidden_state_size in [30, 50, 100, 150]:
    for hidden_state_size in [50]:
        # for lr in [1e-2, 1e-3, 5e-3]:
        for lr in [1e-3]:
            # for batch_size in [32, 64, 128]:
            for batch_size in [100]:
                # for grad_clipping in [None, 0.9]:
                for grad_clipping in [0.9]:
                    print(f'\n\n\nModel num: {counter}, h_s_size: {hidden_state_size}, lr: {lr}, b_size: {batch_size}, g_clip: {grad_clipping}')
                    # counter += 1
                    # if counter < 43:
                    #     continue
                    epochs = 500
                    model, loss = train_AE(lr, batch_size, epochs, hidden_state_size, grad_clipping)
                    if loss < best_loss:
                        best_loss = loss
                        describe_model = (counter, hidden_state_size, lr, batch_size, grad_clipping, loss)
                    # Todo: Check how this model works on the validation set.
                    validation_lost = check_validation(model)

                    # save the model:
                    file_name = f'ae_snp500_{"Adam"}_lr={lr}_hidden_size={hidden_state_size}_gradient_clipping={grad_clipping}_batch_size{batch_size}' \
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

plot_google_amazon_high_stocks()
























