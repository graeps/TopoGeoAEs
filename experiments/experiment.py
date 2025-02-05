import torch
from trainer import EuclideanVAETrainer


class Experiment:
    def __init__(self, config):
        """
        General Experiment class to manage different VAE experiments.

        Args:
            config (dict): Configuration dictionary with experiment parameters.
        """
        self.config = config
        self.trainer = EuclideanVAETrainer(config)

    def run(self):
        """
        Runs the experiment.

        Returns:
            tuple: Training and testing losses.
        """
        train_losses, test_losses = self.trainer.train()
        return train_losses, test_losses
