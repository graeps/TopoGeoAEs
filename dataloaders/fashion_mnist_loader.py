from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_fashion_mnist(batch_size=64, root="../datasets"):
    """
    Loads the Fashion MNIST dataset and returns training and test DataLoaders.

    Args:
        batch_size (int): Number of samples per batch.
        root (str): Directory where data will be downloaded/stored.

    Returns:
        tuple: (train_loader, test_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten the image to a 1D vector
    ])

    train_dataset = datasets.FashionMNIST(root=root, train=True, transform=transform, download=True)
    test_dataset = datasets.FashionMNIST(root=root, train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
