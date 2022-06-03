import numpy as np
import matplotlib.pyplot as plt
import torch


def generate_synth_data(N, T, bach_size=100):
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

    train = torch.tensor(data[:int(N * 0.6), :]).float()
    validation = torch.tensor(data[int(N * 0.6):int(N * 0.8), :]).float()
    test = torch.tensor(data[int(N * 0.8):, :]).float()

    return train.view(int(N * 0.6), T, 1), validation.view(int(N * 0.2), T, 1), test.view(int(N * 0.2), T, 1)


def generate_and_plot():
    np.random.seed(0)  # Constant seed -> Constant results.
    rand_arrs = np.random.rand(10000, 50)  # Generate random arrays.
    for arr in rand_arrs:
        i = np.random.randint(20, 30)
        for ind in range(i - 5, i + 6):
            arr[ind] = arr[ind] * 0.1
    plt.figure()
    xs = np.arange(0, 50, 1)
    for i in range(3):
        ind = np.random.randint(0, 10000)
        plt.plot(xs, rand_arrs[ind], label=f'{ind}')

    plt.xlabel('t')
    plt.ylabel('value')
    plt.title('3 samples from synthetic data-set')
    plt.legend()
    # plt.savefig('samples_from_synthetic_data.png')
    plt.show()
