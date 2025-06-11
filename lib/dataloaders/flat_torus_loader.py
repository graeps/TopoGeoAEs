import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import os


def load_flat_torus_embedding(config):
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "../datasets/flat_torus_50k.pt")

    # Load point cloud tensor
    data = torch.load(file_path).to(torch.float64)
    # Create dummy labels (e.g., all zeros)
    labels = torch.zeros((data.shape[0], 2), dtype=torch.float64)
    # Build dataset
    dataset = TensorDataset(data, labels)

    # Split
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, test_loader

