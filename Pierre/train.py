import torch
import torch.nn as nn
import wandb
import torcheval.metrics as metrics

from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm
from typing import Optional

from pathlib import Path


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        test_dataset: Dataset,
        device: Optional[str] = None,
        batch_size: int = 32,
        lr: float = 0.001,
        model_root: str = "./models",
    ):
        self.device = (
            device if device else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.model_name = model.__class__.__name__
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.lr = lr
        self.model_root = Path(model_root)
        self.model_root.mkdir(parents=True, exist_ok=True)

        # WandB
        self.run = wandb.init(
            project="appml25-final-project",
            entity="plabadens",
            config={
                "batch_size": batch_size,
                "learning_rate": lr,
                "model": self.model_name,
            },
        )

        # Metrics
        self.metrics = {
            "accuracy": metrics.MultilabelAccuracy(),
            "auprc": metrics.MultilabelAUPRC(num_labels=self.model.output_dim),
        }

        # Data
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Model
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.epoch = 0

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

        # Update metrics
        for metric in self.metrics.values():
            metric.update(outputs, Y)

        return loss.item()

    def evaluate(self) -> dict:
        total_loss = 0.0

        with torch.no_grad():
            for X, Y in self.test_loader:
                loss = self._test_step(X, Y)
                total_loss += loss

        # Calculate metrics
        for metric_name, metric in self.metrics.items():
            metric_value = metric.compute()
            self.run.log({metric_name: metric_value})
            metric.reset()

        test_loss = total_loss / len(self.test_loader)

        return test_loss

    def save_model(self):
        artifact_name = f"{self.model_name}-{self.run.id}-e{self.epoch:02d}"

        model_path = self.model_root / f"{artifact_name}.pt"
        torch.save(self.model.state_dict(), model_path)
        self.run.log_artifact(model_path, type="model")

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
                        "train_loss": train_loss,
                        "test_loss": test_loss,
                    }
                )
                self.save_model()

                print(
                    f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}"
                )
                self.epoch += 1

        self.run.finish()


def main():
    from dataset import GalaxyZooDecalsDataset
    from model import EfficientNetZooModel
    import torchvision.transforms.v2 as transforms

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = GalaxyZooDecalsDataset(
        root="./Pierre/dataset", n_rows=1000, transform=transform
    )
    print(f"Dataset loaded with {len(dataset)} samples.")

    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])

    model = EfficientNetZooModel(output_labels=dataset.Y.columns)

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=32,
        lr=0.001,
    )

    trainer.train(epochs=10)


if __name__ == "__main__":
    main()
