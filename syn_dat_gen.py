import numpy as np
import torch
import matplotlib.pyplot as plt


def generate_synth_data(N, T):
    """
    Generate randomly synthetic data.
    :param N: Number of sequences
    :type N: int
    :param T: Length of each sequence
    :type T: int
    :return: Randomly generates synthetic data.
    :rtype: 3-tuple of torch tensors (train, validation, test).
    """
    np.random.seed(0)  # Constant seed -> Constant results.
    rand_arrs = np.random.rand(N, T)  # Generate random arrays.
    for arr in rand_arrs:
        i = np.random.randint(20, 30)
        for ind in range(i - 5, i + 6):
            arr[ind] = arr[ind] * 0.1

    # data = torch.tensor(rand_arrs).view(len(rand_arrs), len(rand_arrs[0]), 1)
    data = rand_arrs
    # Todo: Make sure data is centered and normalized.

    train = torch.tensor(data[:6000, :]).float()
    validation = torch.tensor(data[6000:8000, :]).float()
    test = torch.tensor(data[8000:10000, :]).float()

    return train.view(6000, 50, 1), validation.view(2000, 50, 1), test.view(2000, 50, 1)
