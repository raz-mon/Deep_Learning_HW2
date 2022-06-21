import numpy as np
import torch
from matplotlib import pyplot as plt


def set_seed(seed):
    """Set random seed of the program"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)


def initiate_graph(row, col):
    plt.rc("font", size=12, family="Times New Roman")
    figure, axis = plt.subplots(row, col)
    figure.set_figheight(8)
    figure.set_figwidth(16)
    return axis


def plot_multi_graph(axis, row, col, x, ys, labels, title, x_label, y_label, flag):
    fmt = ["r", "b", "g", "r", "c", "m", "y", "k"]
    if flag:
        for i, y in enumerate(ys):
            axis[row].plot(x, y, fmt[i], label=str(labels[i]))
        axis[row].set_xlabel(x_label, fontdict={"size": 12})
        axis[row].set_ylabel(y_label, fontdict={"size": 12})
        axis[row].set_title(title)
        axis[row].grid(True)
        axis[row].legend()
    else:
        for i, y in enumerate(ys):
            axis[row, col].plot(x, y, fmt[i], label=str(labels[i]))
        axis[row, col].set_xlabel(x_label, fontdict={"size": 12})
        axis[row, col].set_ylabel(y_label, fontdict={"size": 12})
        axis[row, col].set_title(title)
        axis[row, col].grid(True)
        axis[row, col].legend()
