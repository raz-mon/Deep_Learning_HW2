import torch

x = torch.tensor([1, 2, 3])
print(x)
x = x.repeat(1, 5)
print(x)
x = x.view(5, 3)
print(x)
print(x.size())
print(x.size()[0])
















