#
# # This file was not in use in practice - neglect this.
#
#
#
#
#
#
#
#
#
# import torch
# import torch.nn as nn
# import copy
# import numpy as np
# from torch.autograd import Variable
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#
# def generate_synth_data(N, T):
#     """
#     Generate randomly synthetic data.
#     :param N: Number of sequences
#     :type N: int
#     :param T: Length of each sequence
#     :type T: int
#     :return: Randomly generates synthetic data.
#     :rtype: 3-tuple of torch tensors (train, validation, test).
#     """
#     np.random.seed(0)  # Constant seed -> Constant results.
#     rand_arrs = np.random.rand(N, T)  # Generate random arrays.
#     for arr in rand_arrs:
#         i = np.random.randint(20, 30)
#         for ind in range(i - 5, i + 6):
#             arr[ind] = arr[ind] * 0.1
#
#     # data = torch.tensor(rand_arrs).view(len(rand_arrs), len(rand_arrs[0]), 1)
#     data = rand_arrs
#
#     train = torch.tensor(data[:int(N * 0.6), :]).float()
#     validation = torch.tensor(data[int(N * 0.6):int(N * 0.8), :]).float()
#     test = torch.tensor(data[int(N * 0.8):, :]).float()
#
#     return train.view(int(N * 0.6), 50, 1), validation.view(int(N * 0.2), 50, 1), test.view(int(N * 0.2), 50, 1)
#
#
# def train_model(model, train_loader, n_epochs):
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#     criterion = nn.MSELoss().to(device)
#     for epoch in range(1, n_epochs + 1):
#         model = model.train()
#         for data in train_loader:
#             optimizer.zero_grad()
#             data = data.to(device)
#             data_pred = model(data)
#             loss = criterion(data_pred, data)
#             loss.backward()
#             optimizer.step()
#
#
# """"
# class Encoder(nn.Module):
#     def __init__(self, seq_len, n_features):
#         super(Encoder, self).__init__()
#         self.seq_len, self.n_features = seq_len, n_features
#         self.hidden_dim = 2 * n_features
#
#         self.rnn = nn.LSTM(
#             input_size=self.n_features,
#             hidden_size=self.hidden_dim,
#             num_layers=1,
#             batch_first=True
#         )
#
#     def forward(self, x):
#         _, (hidden_n, _) = self.rnn(x)
#         return hidden_n.reshape((self.n_features, self.hidden_dim))
#
#
# class Decoder(nn.Module):
#     def __init__(self, seq_len, n_features=1):
#         super(Decoder, self).__init__()
#         self.seq_len, self.input_dim = seq_len, n_features
#         self.hidden_dim = 2 * n_features
#
#         self.rnn = nn.LSTM(
#             input_size=self.input_dim,
#             hidden_size=self.hidden_dim,
#             num_layers=1,
#             batch_first=True
#         )
#         self.output_layer = nn.Linear(self.hidden_dim, n_features)
#
#     def forward(self, x):
#         x = x.repeat(self.seq_len, self.n_features)
#         x = x.reshape((self.n_features, self.seq_len, self.input_dim))
#         x, (_, _) = self.rnn(x)
#         x = x.reshape((self.seq_len, self.hidden_dim))
#         return self.output_layer(x)
#
#
# class RecurrentAutoencoder(nn.Module):
#     def __init__(self, seq_len, n_features):
#         super(RecurrentAutoencoder, self).__init__()
#         self.encoder = Encoder(seq_len, n_features).to(device)
#         self.decoder = Decoder(seq_len, n_features).to(device)
#
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x
#
#
# model = RecurrentAutoencoder(50, 1)
# model = model.to(device)
# """""
#
#
# class Encoder(nn.Module):
#     def __init__(self, seq_len, n_features, embedding_dim=64):
#         super(Encoder, self).__init__()
#         self.seq_len, self.n_features = seq_len, n_features
#         self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
#         self.rnn1 = nn.LSTM(
#             input_size=n_features,
#             hidden_size=self.hidden_dim,
#             num_layers=1,
#             batch_first=True
#         )
#         self.rnn2 = nn.LSTM(
#             input_size=self.hidden_dim,
#             hidden_size=embedding_dim,
#             num_layers=1,
#             batch_first=True
#         )
#
#     def forward(self, x):
#         x = x.reshape((1, self.seq_len, self.n_features))
#         x, (_, _) = self.rnn1(x)
#         x, (hidden_n, _) = self.rnn2(x)
#         return hidden_n.reshape((self.n_features, self.embedding_dim))
#
#
# class Decoder(nn.Module):
#     def __init__(self, seq_len, input_dim=64, n_features=1):
#         super(Decoder, self).__init__()
#         self.seq_len, self.input_dim = seq_len, input_dim
#         self.hidden_dim, self.n_features = 2 * input_dim, n_features
#         self.rnn1 = nn.LSTM(
#             input_size=input_dim,
#             hidden_size=input_dim,
#             num_layers=1,
#             batch_first=True
#         )
#         self.rnn2 = nn.LSTM(
#             input_size=input_dim,
#             hidden_size=self.hidden_dim,
#             num_layers=1,
#             batch_first=True
#         )
#         self.output_layer = nn.Linear(self.hidden_dim, n_features)
#
#     def forward(self, x):
#         x = x.repeat(self.seq_len, self.n_features)
#         x = x.reshape((self.n_features, self.seq_len, self.input_dim))
#         x, (hidden_n, cell_n) = self.rnn1(x)
#         x, (hidden_n, cell_n) = self.rnn2(x)
#         x = x.reshape((self.seq_len, self.hidden_dim))
#         return self.output_layer(x)
#
#
# class RecurrentAutoencoder(nn.Module):
#     def __init__(self, seq_len, n_features, embedding_dim=64):
#         super(RecurrentAutoencoder, self).__init__()
#         self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
#         self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)
#
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x
#
#
# model = RecurrentAutoencoder(50, 1, 128)
# model = model.to(device)
#
# train_data, val_data, _ = generate_synth_data(100, 50)
# # trainloader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)
#
# train_model(model, train_data, n_epochs=15)
# x = train_data[0]
# print(x)
# print(model.forward(x))
