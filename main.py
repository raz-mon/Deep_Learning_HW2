import torch
import torchvision
import torchvision.transforms as transforms
import util


def main():
    util.set_seed(0)


if __name__ == '__main__':
    main()


def mnist_tests():
    axis = util.initiate_graph(1, 2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((.5,), (.5,))])
    trainset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST('./data', train=False, transform=transform)