"""
import torch
import numpy as np
from torch.utils.data import DataLoader

data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
print(data.shape)
data = torch.tensor(data)
print(data)
data = data.view(4, 3, 1)
print(data)
dataloader = DataLoader(data, batch_size=2, shuffle=False)
for i, dat in enumerate(dataloader):
    print(i, dat, dat.shape)
data = data.detach().numpy()
print(data)
print('yup:', data[1, :])
print(len(np.zeros(len(data))))

# ba = np.array([True, False, True])
# print(ba.sum())







# x = torch.tensor([1, 2, 3])
# print(x)
# x = x.repeat(1, 5)
# print(x)
# x = x.view(5, 3)
# print(x)
# print(x.size())
# print(x.size()[0])

"""














