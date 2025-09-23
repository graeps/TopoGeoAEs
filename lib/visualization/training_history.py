import os

from matplotlib import pyplot as plt


def show_training_history(config, history):
    """
    Visualizes the training history by plotting loss curves for each type of loss
    present in the provided history dictionary. Allows for saving the plotted
    figure to a specified logging directory if `log_dir` is configured in the
    `config` object.

    Args:
        config: An object or configuration with a `log_dir` attribute that
            specifies the directory path to save the training history plot. If set
            to None, the figure will not be saved but only displayed.
        history: A dictionary where keys represent loss types in training (`train_`)
            and validation (`test_`) phases, and values are the respective loss
            values logged per epoch.
    """
    loss_keys = [key.replace('train_', '') for key in history.keys() if key.startswith('train_')]
    unique_losses = sorted(set(loss_keys))

    # Plot setup
    n_losses = len(unique_losses)
    fig, axs = plt.subplots(figsize=(5 * n_losses, 4), ncols=n_losses)

    if n_losses == 1:
        axs = [axs]

    for i, loss_name in enumerate(unique_losses):
        train_key = f'train_{loss_name}'
        test_key = f'test_{loss_name}'
        axs[i].plot(history[train_key], color='orange', label='train')
        axs[i].plot(history[test_key], color='blue', label='val')
        axs[i].set_xlabel('epoch')
        axs[i].set_ylabel('loss')
        axs[i].set_title(f'{loss_name.replace("_", " ").title()} History')
        axs[i].legend()

    fig.subplots_adjust(wspace=0.4)

    if config.log_dir is not None:
        os.makedirs(config.log_dir, exist_ok=True)
        plt.savefig(os.path.join(config.log_dir, "training_history.png"))

    plt.show()

