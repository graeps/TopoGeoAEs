import torch
from ..utils.loss_functions import elbo


class MVAETrainer:
    def __init__(self, model, data_loader, optimizer, config):
        self.config = config
        self.num_epochs = config.num_epochs
        self.log_interval = config.log_interval
        self.device = config.device
        self.recon_loss = config.recon_loss
        self.train_loader, self.test_loader = data_loader
        self.model = model
        self.optimizer = optimizer
        self.history = {'train_loss': [], 'train_recon_loss': [], 'train_kl_loss': [], 'train_topo_loss': [],
                        'test_loss': [],
                        'test_recon_loss': [], 'test_kl_loss': [], 'test_topo_loss': []}
        print("Trainer successfully initialized.")

    def train(self):
        """
        Trains the Euclidean VAE model.

        Returns:
            tuple: Training and testing losses per epoch.
        """
        print("Training the " + f'{self.model.posterior_type}' + "VAE model.")

        for epoch in range(self.num_epochs):
            train_loss, train_recon_loss, train_kl_loss, train_topo_loss = self.train_one_epoch(epoch)
            test_loss, test_recon_loss, test_kl_loss, test_topo_loss = self.test_one_epoch()

            self.history['train_loss'].append(train_loss)
            self.history['train_recon_loss'].append(train_recon_loss)
            self.history['train_kl_loss'].append(train_kl_loss)
            self.history['train_topo_loss'].append(train_topo_loss)
            self.history['test_loss'].append(test_loss)
            self.history['test_recon_loss'].append(test_recon_loss)
            self.history['test_kl_loss'].append(test_kl_loss)
            self.history['test_topo_loss'].append(test_topo_loss)

            print(f"Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
            print("-" * 50)

        return self.history

    def train_one_epoch(self, epoch):
        self.model.train()
        train_loss = 0
        train_recon_loss = 0
        train_kl_loss = 0
        train_topo_loss = 0

        print(f"Starting epoch {epoch + 1}/{self.num_epochs}")

        for batch_idx, (x, _) in enumerate(self.train_loader):
            x = x.to(self.device)
            self.optimizer.zero_grad()
            z, x_recon, posterior_params = self.model(x)
            loss, recon_loss, kl_loss, topo_loss = elbo(self.model.posterior_type, x, z, x_recon, posterior_params,
                                                        self.config)
            loss.backward()

            self.optimizer.step()
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()
            train_topo_loss += topo_loss.item() if hasattr(topo_loss, 'item') else topo_loss

            if (batch_idx + 1) % self.log_interval == 0:
                print(
                    f"Epoch [{epoch + 1}/{self.num_epochs}], Step [{batch_idx + 1}/{len(self.train_loader)}], Loss: {loss.item():.4f}"
                )
        avg_train_loss = train_loss / len(self.train_loader.dataset)
        avg_train_recon_loss = train_recon_loss / len(self.train_loader.dataset)
        avg_train_kl_loss = train_kl_loss / len(self.train_loader.dataset)
        avg_train_topo_loss = train_topo_loss / len(self.train_loader.dataset)
        return avg_train_loss, avg_train_recon_loss, avg_train_kl_loss, avg_train_topo_loss

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
        test_topo_loss = 0

        with torch.no_grad():
            for x, _ in self.test_loader:
                x = x.to(self.device)
                z, x_recon, posterior_params = self.model(x)
                loss, recon_loss, kl_loss, topo_loss = elbo(self.model.posterior_type, x, z, x_recon, posterior_params,
                                                 self.config)
                test_loss += loss.item()
                test_recon_loss += recon_loss.item()
                test_kl_loss += kl_loss.item()
                test_topo_loss += topo_loss.item() if hasattr(topo_loss, 'item') else topo_loss

        avg_test_loss = test_loss / len(self.test_loader.dataset)
        avg_test_recon_loss = test_recon_loss / len(self.test_loader.dataset)
        avg_test_kl_loss = test_kl_loss / len(self.test_loader.dataset)
        avg_test_topo_loss = test_topo_loss / len(self.test_loader.dataset)

        return avg_test_loss, avg_test_recon_loss, avg_test_kl_loss, avg_test_topo_loss
