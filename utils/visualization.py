import matplotlib.pyplot as plt
import seaborn as sns


def plot_losses(train_losses, test_losses):
    """
    Plots training and testing losses over epochs.

    Args:
        train_losses (list): List of training losses per epoch.
        test_losses (list): List of testing losses per epoch.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_reconstruction_errors(errors):
    """
    Plots the distribution of reconstruction errors.

    Args:
        errors (list): List of reconstruction errors.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True, bins=30)
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Reconstruction Errors')
    plt.grid(True)
    plt.show()


def compare_models(results):
    """
    Compares multiple models' performance based on test loss.

    Args:
        results (dict): Dictionary with model names as keys and test losses as values.
    """
    plt.figure(figsize=(10, 6))
    models = list(results.keys())
    losses = list(results.values())
    sns.barplot(x=models, y=losses)
    plt.xlabel('Model')
    plt.ylabel('Test Loss')
    plt.title('Model Performance Comparison')
    plt.grid(True)
    plt.show()
