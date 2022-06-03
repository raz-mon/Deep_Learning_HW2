import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm
import numpy as np
from syn_dat_gen import generate_synth_data
import matplotlib.pyplot as plt


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
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size, bias=False)            # Todo: Do we need this?


    def forward(self, x):
        output, (_, _) = self.encoder.forward(x)                 # z is the last hidden state of the encoder.
        z = output[:, -1].repeat(1, output.shape[1]).view(output.shape)
        z2, (_, _) = self.decoder.forward(z)                # z2 is the last hidden state of the decoder.
        out = self.linear(z2)  # Todo: Do we need this?
        return out


trainset, validationset, testset = generate_synth_data(10000, 50)       # Generate synthetic data.


def train_AE(lr: float, batch_size: int, epochs: int, clip: bool = None):
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    model = LSTM_AE(1, 40)                  # Choosing hidden_state_size to be smaller than the sequence_size, so we won't be learning the id function.
    opt = optim.Adam(model.parameters(), lr)
    critireon = torch.nn.MSELoss()
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


    # save the model:
    # PATH = './lstm_ae_synth.pth'
    # torch.save(model.state_dict(), PATH)  # SAVE THE  WEIGHTS OF THE MODEL (not all of it, can if want, but larger..).

    return model

"""
# Perform grid-search for the best hyper-parameters on the validation-set
def grid_search():
    # validationloader = DataLoader(validationset, batch_size=batch_size, shuffle=False)
    best_hyp_params = []
    last_saved = None
    for hidden_state_size in [10, 50, 100, 150]:
        for lr in [1e-2, 1e-3, 1e-4]:
            for grad_clipping in [0.9, 1.0]:
                # Train model.
                # Test on validation-set. Save hyper-parameters if best result (smallest RMSE?)
                if (last_saved is None) or (is_better(curr_result, last_saved)):
                    best_hyp_params = ['hidden state size': hidden_state_size, 'lr': lr, 'grad clipping': grad_clipping]
"""


def test_validation(model, batch_size):
    # validationloader = DataLoader(validationset, batch_size=validationset.size()[0], shuffle=False)
    loss = torch.nn.MSELoss()
    model.eval()
    output = model(validationset)
    curr_loss = loss(validationset, output)  # print("Accuracy: {:.4f}".format(acc))
    print(f"validation loss = {loss.item()}")
    model.train()


# Todo: Testing = Taking the RMSE?
def test_model(model):
    # testloader = DataLoader(testset, batch_size=testset.size()[0], shuffle=False)
    loss = torch.nn.MSELoss()
    # Test the model on the test-data
    model.eval()  # Change flag in parent model from true to false (train-flag).
    total_loss = 0
    with torch.no_grad():  # Everything below - will not calculate the gradients.
        # for data in testset:
        # Apply model (forward pass).
        outputs = model(testset)

        total_loss += loss(testset, outputs)                    # MSELoss of the output and data


AE = train_AE(1e-3, 500, 400)
test_model(AE)
# print a ts and a reconstruction of it.
xs = np.arange(0, 50, 1)
ys1 = testset[0, :, :]
ys2 = AE(ys1).view(50).detach().numpy()
plt.plot(xs, ys1.view(50).detach().numpy(), label='orig')
plt.plot(xs, ys2, label='rec')
plt.legend()
plt.title('original and reconstructed signal')
plt.show()












