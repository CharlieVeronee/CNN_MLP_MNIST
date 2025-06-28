from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#loads mnist data from pytorch library
#add the transform ToTensor() when formatting the dataset, to convert the input data from a Pillow Image type into a PyTorch Tensor (a multi-dimensional array)
#split into train and test data with DataLoader

def load_data(batch_size=100):
    transform = transforms.ToTensor()
    mnist_train = datasets.MNIST(root="./datasets", train=True, transform=transform, download=True)
    mnist_test = datasets.MNIST(root="./datasets", train=False, transform=transform, download=True)

    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader