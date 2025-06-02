import torch
import torch.nn as nn
import wandb

from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from typing import Optional


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        test_dataset: Dataset,
        device: Optional[str] = None,
        batch_size: int = 32,
        lr: float = 0.001,
    ):
        self.device = (
            device if device else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.lr = lr

        # WandB
        self.run = wandb.init(
            project="appml25-final-project",
            entity="plabadens",
            config={
                "batch_size": batch_size,
                "learning_rate": lr,
                "model": model.__class__.__name__,
            },
        )

        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def _train_step(self, X, Y):
        self.model.train()
        X = X.to(self.device)
        Y = Y.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(X)
        loss = self.criterion(outputs, Y)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _test_step(self, X, Y):
        self.model.eval()
        X = X.to(self.device)
        Y = Y.to(self.device)

        with torch.inference_mode():
            outputs = self.model(X)
            loss = self.criterion(outputs, Y)

        return loss.item()

    def evaluate(self):
        total_loss = 0.0
        with torch.no_grad():
            for X, Y in self.test_loader:
                loss = self._test_step(X, Y)
                total_loss += loss

        return total_loss / len(self.test_loader)

    def train(self, epochs: int = 10):
        total_steps = self.batch_size * len(self.train_loader) * epochs

        with tqdm(
            total=total_steps,
            desc="Training",
            unit="img",
        ) as progress_bar:
            for epoch in range(epochs):
                train_loss = 0.0
                for X, Y in self.train_loader:
                    loss = self._train_step(X, Y)
                    train_loss += loss
                    progress_bar.update(self.batch_size)

                train_loss /= len(self.train_loader)
                progress_bar.set_postfix({"loss": train_loss})

                test_loss = self.evaluate()
                progress_bar.set_postfix({"test_loss": test_loss})

                self.run.log(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "test_loss": test_loss,
                    }
                )
                print(
                    f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}"
                )
