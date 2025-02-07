import matplotlib.pyplot as plt


def plot_losses(train_losses, test_losses, num_epochs):
    """
    Plot training and testing losses over epochs.

    Parameters
    ----------
    train_losses : list of float
        List of training losses, where each element represents the loss for a specific epoch.
    test_losses : list of float
        List of testing losses, where each element represents the loss for a specific epoch.
    num_epochs : int
        Total number of epochs for which the losses are recorded.

    Returns
    -------
    None
    """
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss", marker="o")
    plt.plot(range(1, num_epochs + 1), test_losses, label="Test Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train Loss vs Test Loss")
    plt.legend()
    plt.grid(True)
    plt.show()
