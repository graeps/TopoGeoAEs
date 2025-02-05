import torch
from loss_functions import euclid_gaussian_loss


class Evaluation:
    def __init__(self, model, data_loader, device):
        """
        Evaluation class for assessing model performance.

        Args:
            model (torch.nn.Module): The model to evaluate.
            data_loader (DataLoader): DataLoader for the evaluation dataset.
            device (torch.device): Device to run the evaluation on.
        """
        self.model = model
        self.data_loader = data_loader
        self.device = device

    def evaluate(self):
        """
        Evaluates the model on the provided dataset.

        Returns:
            float: Average evaluation loss.
        """
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for x, _ in self.data_loader:
                x = x.to(self.device)
                reconstructed, mu, logvar = self.model(x)
                loss, _, _ = euclid_gaussian_loss(reconstructed, x, mu, logvar)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.data_loader.dataset)
        return avg_loss

    def compute_reconstruction_error(self):
        """
        Computes the reconstruction error for the dataset.

        Returns:
            list: Reconstruction errors for each batch.
        """
        self.model.eval()
        errors = []

        with torch.no_grad():
            for x, _ in self.data_loader:
                x = x.to(self.device)
                reconstructed, _, _ = self.model(x)
                error = torch.mean((x - reconstructed) ** 2, dim=[1, 2, 3])
                errors.extend(error.cpu().numpy())

        return errors
