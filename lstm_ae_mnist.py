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
    def __init__(self, input_size, hidden_size):
        super(LSTM_AE_MNIST, self).__init__()
        # Regular Parameters
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Torch parameters
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.reconstruct_linear = nn.Linear(hidden_size, input_size, bias=False)  # Todo: Do we need this?
        self.classify_linear = nn.Linear(hidden_size, 10, bias=False)

    def forward(self, x):
        output, (_, _) = self.encoder.forward(x)  # z is the last hidden state of the encoder.
        z = output[:, -1].repeat(1, output.shape[1]).view(output.shape)
        z2, (_, _) = self.decoder.forward(z)  # z2 is the last hidden state of the decoder.
        return self.reconstruct_linear(z2), self.classify_linear(z2[:, -1])


def reshape_row_by_row(data, batch_size):
    data = data.clone().detach()
    return data.view(batch_size, 28, 28)


def reshape_pixel_by_pixel(data, batch_size):
    data = data.clone().detach()
    return data.view(batch_size, 28 * 28, 1)


def train_AE(input, lr: float, batch_size: int, epochs: int, hidden_size, clip, is_reconstruct, form):
    format_dict = {"rbr": lambda x: reshape_row_by_row(x, batch_size),
               "pbp": lambda x: reshape_pixel_by_pixel(x, batch_size)}
    convert = format_dict[form]
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    model = LSTM_AE_MNIST(input, hidden_size)
    # Choosing hidden_state_size to be smaller than the sequence_size, so we won't be learning the id function.
    opt = optim.Adam(model.parameters(), lr)
    criterion = torch.nn.MSELoss()
    CE = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    best_loss = float('inf')
    best_epoch = 0
    for epoch in range(epochs):
        total_loss = 0.0
        for i, (data, target) in enumerate(trainloader):
            print(i)
            opt.zero_grad()
            data = data.to(device)
            target = target.to(device)
            data = convert(data)
            output_reconstruct, output_classify = model(data)
            if is_reconstruct:
                loss = criterion(data, output_reconstruct)
            else:
                #temp = torch.tensor(np.argmax(output_classify.detach().numpy(), axis=1))
                loss = criterion(data, output_reconstruct) + CE(output_classify, target)
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
                f'_epoch{epochs}_best_epoch{best_epoch}_best_loss{best_loss}_isreconstruct{is_reconstruct}'

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
        for lr in [1e-2, 1e-3, 5e-3]:
            for batch_size in [32, 64, 128]:
                for grad_clipping in [None, 0.9]:
                    print(
                        f'\n\n\nModel num: {counter}, h_s_size: {hidden_state_size}, lr: {lr}, b_size: {batch_size}, g_clip: {grad_clipping}')
                    counter += 1
                    if counter > 1:
                        break
                    _, loss = train_AE(28, lr, batch_size, 3, hidden_state_size, grad_clipping, False, "rbr")
                    if loss < best_loss:
                        best_loss = loss
                        describe_model = (counter, hidden_state_size, lr, batch_size, grad_clipping, loss)
    print(
        "best model {} params:\nhidden state: {}\nlearning state: {}\nbatch size: {}\ngrad clipping: {}\nloss: {}".format(
            *describe_model))


# Todo: Testing = Taking the RMSE?
def test_model(model):
    # testloader = DataLoader(testset, batch_size=testset.size()[0], shuffle=False)
    loss = torch.nn.MSELoss()
    # Test the model on the test-data
    model.eval()  # Change flag in parent model from true to false (train-flag).
    total_loss = 0
    # with torch.no_grad():  # Everything below - will not calculate the gradients.
    # for data in testset:
    # Apply model (forward pass).
    # outputs = model(testset)

    # total_loss += loss(testset, outputs)  # MSELoss of the output and data


def imshow(img):
    plt.imshow(img[0])
    plt.show()


set_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((.5,), (.5,))])

trainset = torchvision.datasets.MNIST('./mnist_data/', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST('./mnist_data/', train=False, download=True, transform=transform)

# grid_search()
# model = torch.load("saved_models/mnist_task/ae_mnist_Adam_lr=0.01_hidden_size=30_gradient_clipping=None_batch_size32_epoch2_best_epoch1_best_loss310.1443293467164_isreconstructTrue.pt")
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
# for i, (data, target) in enumerate(trainloader):
#     if not i:
#         imshow(data[0])
#         data = reshape_row_by_row(data, 1)
#         new_data, _ = model(data)
#         imshow(new_data.detach().numpy())
#     break

grid_search()

"""
model = torch.load("saved_models/mnist_task/ae_mnist_Adam_lr=0.01_hidden_size=30_gradient_clipping=None_batch_size32_epoch3_best_epoch2_best_loss1098.8780343830585_isreconstructFalse.pt")
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
for i, (data, target) in enumerate(trainloader):
    if not i:
        imshow(data[0])
        data = reshape_row_by_row(data, 1)
        new_data, cls = model(data)
        x = torch.tensor(np.argmax(cls.detach().numpy(), axis=1))
        imshow(new_data.detach().numpy())
    break
"""
