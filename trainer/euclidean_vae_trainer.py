import torch
import torch.optim as optim
from ..models.euclidean_vae import EuclideanVAE
from ..utils.loss_functions import euclid_gaussian_loss
from ..utils.evaluation import evaluate_model


class EuclideanVAETrainer:
    def __init__(self, config, data_loader):
        """
        Trainer class for the Euclidean Variational Autoencoder (VAE).

        Args:
            config (dict): Configuration dictionary with training parameters.
            data_loader (tuple): Tuple containing (train_loader, test_loader).
        """
        self.latent_dim = config.get('latent_dim', 20)
        self.batch_size = config.get('batch_size', 64)
        self.num_epochs = config.get('num_epochs', 10)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.device = config.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        self.train_loader, self.test_loader = data_loader
        self.model = EuclideanVAE(self.latent_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.train_losses = []
        self.test_losses = []

    def train(self):
        """
        Trains the Euclidean VAE model.

        Returns:
            tuple: Training and testing losses per epoch.
        """
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0

            for x, _ in self.train_loader:
                x = x.to(self.device)
                self.optimizer.zero_grad()
                reconstructed, mu, logvar = self.model(x)
                loss, _, _ = loss_function(reconstructed, x, mu, logvar)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(self.train_loader.dataset)
            self.train_losses.append(avg_train_loss)

            test_loss = self.evaluate()

            print(f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}")

        return self.train_losses, self.test_losses

    def evaluate(self):
        """
        Evaluates the Euclidean VAE model on the test dataset.

        Returns:
            float: Average test loss.
        """
        avg_test_loss = evaluate_model(self.model, self.test_loader, self.device)
        self.test_losses.append(avg_test_loss)

        return avg_test_loss
