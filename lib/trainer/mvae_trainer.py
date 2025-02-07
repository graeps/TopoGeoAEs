import torch
from ..utils.loss_functions import elbo


class MVAETrainer:
    def __init__(self, model, data_loader, optimizer, config):
        self.num_epochs = config.get('num_epochs', 10)
        self.log_interval = config.get('log_interval', 100)
        self.device = config.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.train_loader, self.test_loader = data_loader
        self.model = model
        self.optimizer = optimizer
        self.train_losses = []
        self.test_losses = []
        print("Trainer successfully initialized.")

    def train(self):
        """
        Trains the Euclidean VAE model.

        Returns:
            tuple: Training and testing losses per epoch.
        """
        print("Training the" + f'{self.model.posterior_type}' + "VAE model.")
        for epoch in range(self.num_epochs):
            avg_train_loss = self.train_one_epoch(epoch)

            self.train_losses.append(avg_train_loss)

            test_loss = self.test_one_epoch()

            print(f"Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}")
            print("-" * 50)

        return self.train_losses, self.test_losses

    def train_one_epoch(self, epoch):
        self.model.train()
        train_loss = 0

        print(f"Starting epoch {epoch + 1}/{self.num_epochs}")

        for batch_idx, (x, _) in enumerate(self.train_loader):
            x = x.to(self.device)
            self.optimizer.zero_grad()
            x_recon, posterior_params = self.model(x)
            loss, _, _ = elbo(self.model.posterior_type, x, x_recon, posterior_params, self.model.latent_dim,
                              self.device)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

            if (batch_idx + 1) % self.log_interval == 0:
                print(
                    f"Epoch [{epoch + 1}/{self.num_epochs}], Step [{batch_idx + 1}/{len(self.train_loader)}], Loss: {loss.item():.4f}"
                )
        train_loss /= len(self.train_loader.dataset)
        return train_loss

    def test_one_epoch(self):
        """
        Evaluates the Euclidean VAE model on the test dataset.

        Returns:
            float: Average test loss.
        """
        self.model.eval()
        test_loss = 0

        with torch.no_grad():
            for x, _ in self.test_loader:
                x = x.to(self.device)
                x_recon, posterior_params = self.model(x)
                loss, _, _ = elbo(self.model.posterior_type, x, x_recon, posterior_params, self.model.latent_dim,
                                  self.device)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(self.test_loader.dataset)
        self.test_losses.append(avg_test_loss)

        return avg_test_loss
