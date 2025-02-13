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
        self.history = {'train_loss': [], 'train_recon_loss': [], 'train_kl_loss': [], 'test_loss': [],
                        'test_recon_loss': [], 'test_kl_loss': []}
        print("Trainer successfully initialized.")

    def train(self):
        """
        Trains the Euclidean VAE model.

        Returns:
            tuple: Training and testing losses per epoch.
        """
        print("Training the " + f'{self.model.posterior_type}' + "VAE model.")

        for epoch in range(self.num_epochs):
            train_loss, train_recon_loss, train_kl_loss = self.train_one_epoch(epoch)
            test_loss, test_recon_loss, test_kl_loss = self.test_one_epoch()

            self.history['train_loss'].append(train_loss)
            self.history['train_recon_loss'].append(train_recon_loss)
            self.history['train_kl_loss'].append(train_kl_loss)
            self.history['test_loss'].append(test_loss)
            self.history['test_recon_loss'].append(test_recon_loss)
            self.history['test_kl_loss'].append(test_kl_loss)

            print(f"Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
            print("-" * 50)

        return self.history

    def train_one_epoch(self, epoch):
        self.model.train()
        train_loss = 0
        train_recon_loss = 0
        train_kl_loss = 0

        print(f"Starting epoch {epoch + 1}/{self.num_epochs}")

        for batch_idx, (x, _) in enumerate(self.train_loader):
            x = x.to(self.device)
            self.optimizer.zero_grad()
            _, x_recon, posterior_params = self.model(x)
            loss, recon_loss, kl_loss = elbo(self.model.posterior_type, x, x_recon, posterior_params,
                                             self.model.latent_dim,
                                             self.device)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()

            if (batch_idx + 1) % self.log_interval == 0:
                print(
                    f"Epoch [{epoch + 1}/{self.num_epochs}], Step [{batch_idx + 1}/{len(self.train_loader)}], Loss: {loss.item():.4f}"
                )
        avg_train_loss = train_loss / len(self.train_loader.dataset)
        avg_train_recon_loss = train_recon_loss / len(self.train_loader.dataset)
        avg_train_kl_loss = train_kl_loss / len(self.train_loader.dataset)
        return avg_train_loss, avg_train_recon_loss, avg_train_kl_loss

    def test_one_epoch(self):
        """
        Evaluates the Euclidean VAE model on the test dataset.

        Returns:
            float: Average test loss.
        """
        self.model.eval()
        test_loss = 0
        test_recon_loss = 0
        test_kl_loss = 0

        with torch.no_grad():
            for x, _ in self.test_loader:
                x = x.to(self.device)
                _, x_recon, posterior_params = self.model(x)
                loss, recon_loss, kl_loss = elbo(self.model.posterior_type, x, x_recon, posterior_params,
                                                 self.model.latent_dim,
                                                 self.device)
                test_loss += loss.item()
                test_recon_loss += recon_loss.item()
                test_kl_loss += kl_loss.item()

        avg_test_loss = test_loss / len(self.test_loader.dataset)
        avg_test_recon_loss = test_recon_loss / len(self.test_loader.dataset)
        avg_test_kl_loss = test_kl_loss / len(self.test_loader.dataset)

        return avg_test_loss, avg_test_recon_loss, avg_test_kl_loss
