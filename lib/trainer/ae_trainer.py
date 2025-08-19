import torch
from tqdm import tqdm
from ..utils.loss_functions import topo_ae_loss


class AETrainer:
    def __init__(self, model, data_loader, optimizer, config):
        self.config = config
        self.num_epochs = config.num_epochs
        self.log_interval = config.log_interval
        self.verbose = config.verbose
        self.device = config.device
        self.recon_loss = config.recon_loss
        self.train_loader, self.test_loader = data_loader
        self.model = model
        self.optimizer = optimizer
        self.history = {'train_loss': [], 'train_recon_loss': [], 'train_topo_loss': [], 'test_loss': [],
                        'test_recon_loss': [], 'test_topo_loss': []}

        if config.verbose:
            print("Trainer successfully initialized.")

    def train(self):
        if self.verbose:
            print(f"Training the {self.model.type} AE model.")

        for epoch in range(self.num_epochs):
            train_loss, train_recon_loss, train_topo_loss = self.train_one_epoch(epoch, self.verbose)
            test_loss, test_recon_loss, test_topo_loss = self.test_one_epoch()

            self.history['train_loss'].append(train_loss)
            self.history['train_recon_loss'].append(train_recon_loss)
            self.history['train_topo_loss'].append(train_topo_loss)
            self.history['test_loss'].append(test_loss)
            self.history['test_recon_loss'].append(test_recon_loss)
            self.history['test_topo_loss'].append(test_topo_loss)

            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
                print("-" * 50)

        return self.history

    def train_one_epoch(self, epoch, verbose):
        self.model.train()
        train_loss = 0
        train_recon_loss = 0
        train_topo_loss = 0

        if verbose:
            print(f"Starting epoch {epoch + 1}/{self.num_epochs}")
            dataloader = self.train_loader
        else:
            dataloader = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}", leave=False)

        for batch_idx, (x, _) in enumerate(dataloader):
            x = x.to(self.device)
            self.optimizer.zero_grad()
            angles, z, x_recon = self.model(x)

            loss, recon_loss, topo_loss = topo_ae_loss(self.config, x, z, x_recon)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_topo_loss += topo_loss.item() if hasattr(topo_loss, 'item') else topo_loss

            if ((batch_idx + 1) % self.log_interval == 0) and verbose:
                print(
                    f"Epoch [{epoch + 1}/{self.num_epochs}], Batch [{batch_idx + 1}/{len(self.train_loader)}], Loss: {loss.item():.4f}")

        n_samples = len(self.train_loader.dataset)
        return (train_loss / n_samples,
                train_recon_loss / n_samples,
                train_topo_loss / n_samples)

    def test_one_epoch(self):
        """
        Evaluates the Euclidean VAE model on the test dataset.

        Returns:
            float: Average test loss.
        """
        self.model.eval()
        test_loss = 0
        test_recon_loss = 0
        test_topo_loss = 0

        with torch.no_grad():
            for x, _ in self.test_loader:
                x = x.to(self.device)
                angles, z, x_recon = self.model(x)
                loss, recon_loss, topo_loss = topo_ae_loss(self.config, x, z, x_recon)
                test_loss += loss.item()
                test_recon_loss += recon_loss.item()
                test_topo_loss += topo_loss.item() if hasattr(topo_loss, 'item') else topo_loss

        avg_test_loss = test_loss / len(self.test_loader.dataset)
        avg_test_recon_loss = test_recon_loss / len(self.test_loader.dataset)
        avg_test_topo_loss = test_topo_loss / len(self.test_loader.dataset)

        return avg_test_loss, avg_test_recon_loss, avg_test_topo_loss
