import torch


class AETrainer:
    def __init__(self, model, data_loader, optimizer, config):
        self.num_epochs = config.get('num_epochs', 10)
        self.log_interval = config.get('log_interval', 100)
        self.device = config.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.train_loader, self.test_loader = data_loader
        self.model = model
        self.optimizer = optimizer
        self.recon_loss = torch.nn.MSELoss()
        self.history = {'train_recon_loss': [], 'test_recon_loss': []}
        print("Trainer successfully initialized.")

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch_idx, (x, _) in enumerate(self.train_loader):
            x = x.to(self.device)
            _, x_recon = self.model(x)
            loss = self.recon_loss(x_recon, x)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % self.log_interval == 0:
                print(
                    f"Step [{batch_idx + 1}/{len(self.train_loader)}], Loss: {loss.item():.4f}"
                )
        return total_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, _ in self.test_loader:
                x = x.to(self.device)
                _, x_recon = self.model(x)
                loss = self.recon_loss(x_recon, x)
                total_loss += loss.item()
        return total_loss / len(self.test_loader)

    def train(self):
        print("Start training...")
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch}")
            train_loss = self.train_epoch()
            test_loss = self.evaluate()
            print(f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
