import torch
from tqdm import tqdm
from ..utils.loss_functions import elbo


class MVAETrainer:
    """
    Generic trainer for manifold-based VAEs (MVAE).

    Handles training and evaluation loops, including:
    - Forward passes through the model
    - ELBO computation (reconstruction + KL + optional topological loss)
    - Parameter updates via the provided optimizer
    - Epoch-wise aggregation of loss statistics
    """

    def __init__(self, model, data_loader, optimizer, config):
        """
        Initialize the MVAETrainer.

        Args:
            model (torch.nn.Module): VAE model to train.
            data_loader (Tuple[DataLoader, DataLoader]): Tuple of (train_loader, test_loader).
            optimizer (torch.optim.Optimizer): Optimizer for model parameters.
            config: Configuration object with required attributes:
                * num_epochs (int): Number of training epochs.
                * log_interval (int): Logging frequency in batches.
                * verbose (bool): If True, print training progress.
                * device (torch.device): Device to run computations on.
                * recon_loss (str): Type of reconstruction loss (e.g., 'mse' or 'bce').
        """
        self.config = config
        self.num_epochs = config.num_epochs
        self.log_interval = config.log_interval
        self.verbose = config.verbose
        self.device = config.device
        self.recon_loss = config.recon_loss

        # Data
        self.train_loader, self.test_loader = data_loader

        # Model and optimizer
        self.model = model
        self.optimizer = optimizer

        # Training history
        self.history = {
            'train_loss': [],
            'train_recon_loss': [],
            'train_kl_loss': [],
            'train_topo_loss': [],
            'test_loss': [],
            'test_recon_loss': [],
            'test_kl_loss': [],
            'test_topo_loss': []
        }

        if self.verbose:
            print("Trainer successfully initialized.")

    def train(self):
        """
        Train the VAE model for a fixed number of epochs.

        Returns:
            dict: History of losses with keys:
                * 'train_loss', 'test_loss' (float): Total losses per epoch.
                * 'train_recon_loss', 'test_recon_loss' (float): Reconstruction losses.
                * 'train_kl_loss', 'test_kl_loss' (float): KL divergence terms.
                * 'train_topo_loss', 'test_topo_loss' (float): Topological losses if present.
        """
        if self.verbose:
            print(f"Training the {self.model.type} VAE model.")

        for epoch in range(self.num_epochs):
            # Train on one epoch and compute metrics
            train_loss, train_recon_loss, train_kl_loss, train_topo_loss = self.train_one_epoch(epoch, self.verbose)

            # Evaluate on test set
            test_loss, test_recon_loss, test_kl_loss, test_topo_loss = self.test_one_epoch()

            # Store losses
            self.history['train_loss'].append(train_loss)
            self.history['train_recon_loss'].append(train_recon_loss)
            self.history['train_kl_loss'].append(train_kl_loss)
            self.history['train_topo_loss'].append(train_topo_loss)
            self.history['test_loss'].append(test_loss)
            self.history['test_recon_loss'].append(test_recon_loss)
            self.history['test_kl_loss'].append(test_kl_loss)
            self.history['test_topo_loss'].append(test_topo_loss)

            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.num_epochs}, "
                      f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
                print("-" * 50)

        return self.history

    def train_one_epoch(self, epoch, verbose):
        """
        Perform a single epoch of training.

        Args:
            epoch (int): Current epoch index.
            verbose (bool): If True, print batch-wise progress.

        Returns:
            Tuple[float, float, float, float]:
                Average total loss, reconstruction loss, KL loss,
                and topological loss for the epoch.
        """
        self.model.train()
        train_loss = 0.0
        train_recon_loss = 0.0
        train_kl_loss = 0.0
        train_topo_loss = 0.0

        # Use tqdm progress bar if not verbose
        dataloader = self.train_loader if verbose else tqdm(
            self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}", leave=False
        )

        for batch_idx, (x, labels) in enumerate(dataloader):
            x = x.to(self.device)
            self.optimizer.zero_grad()

            # Forward pass
            z, x_recon, posterior_params = self.model(x)

            # Compute ELBO and components
            loss, recon_loss, kl_loss, topo_loss = elbo(
                self.model.type, x, z, x_recon, posterior_params, self.config
            )

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            # Accumulate batch losses
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()
            train_topo_loss += topo_loss.item() if hasattr(topo_loss, 'item') else topo_loss

            if ((batch_idx + 1) % self.log_interval == 0) and verbose:
                print(
                    f"Epoch [{epoch + 1}/{self.num_epochs}], "
                    f"Batch [{batch_idx + 1}/{len(self.train_loader)}], "
                    f"Loss: {loss.item():.4f}"
                )

        n_samples = len(self.train_loader.dataset)
        return (
            train_loss / n_samples,
            train_recon_loss / n_samples,
            train_kl_loss / n_samples,
            train_topo_loss / n_samples
        )

    def test_one_epoch(self):
        """
        Evaluate the VAE on the test dataset.

        Returns:
            Tuple[float, float, float, float]:
                Average total loss, reconstruction loss, KL loss,
                and topological loss on the test set.
        """
        self.model.eval()
        test_loss = 0.0
        test_recon_loss = 0.0
        test_kl_loss = 0.0
        test_topo_loss = 0.0

        with torch.no_grad():
            for x, labels in self.test_loader:
                x = x.to(self.device)

                # Forward pass
                z, x_recon, posterior_params = self.model(x)

                # Compute ELBO and components
                loss, recon_loss, kl_loss, topo_loss = elbo(
                    self.model.type, x, z, x_recon, posterior_params, self.config
                )

                # Accumulate losses
                test_loss += loss.item()
                test_recon_loss += recon_loss.item()
                test_kl_loss += kl_loss.item()
                test_topo_loss += topo_loss.item() if hasattr(topo_loss, 'item') else topo_loss

        n_samples = len(self.test_loader.dataset)
        return (
            test_loss / n_samples,
            test_recon_loss / n_samples,
            test_kl_loss / n_samples,
            test_topo_loss / n_samples
        )
