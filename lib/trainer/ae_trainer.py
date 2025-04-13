import torch
import matplotlib.pyplot as plt
import numpy as np


class AETrainer:
    def __init__(self, model, data_loader, optimizer, config):
        self.num_epochs = config.get('num_epochs', 10)
        self.log_interval = config.get('log_interval', 100)
        self.device = config.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.dataset = config.get("dataset", "MNIST")
        self.show_latents = config.get("show_latents", False)
        self.train_loader, self.test_loader = data_loader
        self.model = model
        self.optimizer = optimizer
        self.recon_loss = torch.nn.MSELoss()
        self.history = {'train_recon_loss': [], 'test_recon_loss': []}
        print("Trainer successfully initialized.")

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        latent_angles = []

        if self.dataset == "synthetic":
            for batch_idx, x in enumerate(self.train_loader):
                x = x[0].to(self.device)
                if self.model.type == "euclidean_ae":
                    z, x_recon = self.model(x)
                    A = "not defined"
                    A_inv_T = "not defined"
                elif self.model.type == "shape_toroidal_ae":
                    theta, x_recon, (A, A_inv_T), theta = self.model(x)
                    latent_angles.append(theta)
                else:
                    raise ValueError(f"Unknown model type: {self.model.type}")

                loss = self.recon_loss(x_recon, x)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                if (batch_idx + 1) % self.log_interval == 0:
                    print(
                        f"Step [{batch_idx + 1}/{len(self.train_loader)}], Loss: {loss.item():.4f}, Shape matrix A:{A}, A_inv_T:{A_inv_T}"
                    )
            if self.model.type == "shape_toroidal_ae" and self.show_latents:
                plot_angles(latent_angles)

            return total_loss / len(self.train_loader)

        else:
            for batch_idx, (x, _) in enumerate(self.train_loader):
                x = x.to(self.device)
                if self.model.type == "euclidean_ae":
                    z, x_recon = self.model(x)
                    A = "not defined"
                    A_inv_T = "not defined"
                elif self.model.type == "shape_toroidal_ae":
                    theta, x_recon, (A, A_inv_T), theta = self.model(x)
                else:
                    raise ValueError(f"Unknown model type: {self.model.type}")
                loss = self.recon_loss(x_recon, x)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                if (batch_idx + 1) % self.log_interval == 0:
                    print(
                        f"Step [{batch_idx + 1}/{len(self.train_loader)}], Loss: {loss.item():.4f}, Shape matrix A:{A}, A_inv_T:{A_inv_T}"
                    )
            return total_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x in self.test_loader:
                if self.dataset == "synthetic":
                    x = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                if self.model.type == "euclidean_ae":
                    z, x_recon = self.model(x)
                    A = "not defined"
                    A_inv_T = "not defined"
                elif self.model.type == "shape_toroidal_ae":
                    theta, x_recon, A, A_inv_T = self.model(x)
                loss = self.recon_loss(x_recon, x)
                total_loss += loss.item()
        return total_loss / len(self.test_loader)

    def train(self):
        print("Start training...")
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch}")
            train_loss = self.train_epoch()
            test_loss = self.evaluate()
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")



def plot_angles(latent_angles):
    latent_angles = torch.cat(latent_angles, dim=0).detach().numpy()

    if latent_angles.shape[1] == 1:  # Case d = 1
        fig, ax = plt.subplots(figsize=(5, 3))
        theta = latent_angles[:, 0]
        ax.scatter(theta, np.arange(len(theta)), s=2)
        ax.set_title("Latent Angles (1D)")
        ax.set_xlabel("θ")
        ax.set_ylabel("Index")
        ax.grid(True, linestyle='--', alpha=0.5)

    elif latent_angles.shape[1] == 2:  # Case d = 2
        fig, ax = plt.subplots(figsize=(5, 5))
        theta1 = latent_angles[:, 0]
        theta2 = latent_angles[:, 1]
        ax.scatter(theta1, theta2, s=2)
        ax.set_title("Latent Angles (2D)")
        ax.set_xlabel("θ_1")
        ax.set_ylabel("θ_2")
        ax.set_aspect('equal', adjustable='datalim')
        ax.autoscale()
        ax.grid(True, linestyle='--', alpha=0.5)

    else:
        raise ValueError(f"Unsupported latent dimension: {latent_angles.shape[1]}")

    plt.show()
