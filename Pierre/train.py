import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp as amp
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
        batch_size: int = 32,
        device: Optional[str] = None,
        epochs: int = 10,
        lr: float = 1e-4,
        lr_scheduler: Optional[str] = None,
        weight_decay: float = 1e-3,
        freeze_blocks: int = 0,
        model_root: str = "./models",
        train_transform: Optional[nn.Module] = None,
        problem_type: str = "multiclass",
        use_amp: bool = True,
    ):
        """
        Initializes the Trainer with the model, datasets, and hyperparameters.

        Args:
            model (nn.Module): The model to be trained.
            train_dataset (Dataset): The training dataset.
            test_dataset (Dataset): The testing dataset.
            batch_size (int): Batch size for training and evaluation.
            device (Optional[str]): Device to run the model on ("cuda" or "cpu").
            epochs (int): Number of epochs to train the model.
            lr (float): Learning rate for the optimizer.
            lr_sheduler (Optional[str]): Type of learning rate scheduler to use (e.g., "cosine").
            weight_decay (float): Weight decay for the optimizer.
            freeze_blocks (int): Number of blocks to freeze in the model.
            model_root (str): Directory to save the trained models.
            train_transform (Optional[nn.Module]): Transformations to apply to training data.
            problem_type (str): Type of problem ("multiclass" or "multilabel").
            use_amp (bool): Whether to use automatic mixed precision training.
        """

        if problem_type not in ["multiclass", "multilabel"]:
            raise ValueError(f"Unsupported problem type: {problem_type}")

        self.device = (
            device if device else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.model_name = model.__class__.__name__
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model_root = Path(model_root)
        self.model_root.mkdir(parents=True, exist_ok=True)

        # Hyperparameters
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs

        # WandB
        self.run = wandb.init(
            project="appml25-final-project",
            entity="plabadens",
            config={
                "batch_size": batch_size,
                "learning_rate": lr,
                "learning_rate_scheduler": lr_scheduler,
                "model": self.model_name,
                "weight_decay": weight_decay,
                "train_size": len(train_dataset),
                "test_size": len(test_dataset),
                "problem_type": problem_type,
                "freeze_blocks": freeze_blocks,
            },
        )

        # Metrics
        if problem_type == "multiclass":
            self.metrics = {
                "accuracy": metrics.MulticlassAccuracy(
                    num_classes=self.model.output_dim
                ),
                "precision": metrics.MulticlassPrecision(
                    num_classes=self.model.output_dim
                ),
                "recall": metrics.MulticlassRecall(num_classes=self.model.output_dim),
                "f1_score": metrics.MulticlassF1Score(
                    num_classes=self.model.output_dim
                ),
            }
        elif problem_type == "multilabel":
            self.metrics = {
                "accuracy": metrics.MultilabelAccuracy(),
                "auprc": metrics.MultilabelAUPRC(num_labels=self.model.output_dim),
                "top3_accuracy": metrics.TopKMultilabelAccuracy(k=3),
            }

        # Data
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

        # Transforms
        self.train_transform = train_transform

        # Model
        if freeze_blocks > 0:
            self.model.freeze_blocks(freeze_blocks)

        if problem_type == "multiclass":
            self.criterion = nn.CrossEntropyLoss()
        elif problem_type == "multilabel":
            self.criterion = nn.BCEWithLogitsLoss()

        self.use_amp = use_amp
        self.scaler = amp.GradScaler(enabled=self.use_amp)

        self.optimizer = self._create_optimizer()

        if lr_scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs * len(self.train_loader), eta_min=0.0
            )
        elif lr_scheduler == "cosine_warmup":
            total_steps = epochs * len(self.train_loader)
            warmup_steps = int(0.1 * total_steps)

            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1e-3, total_iters=warmup_steps
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps - warmup_steps
            )

            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )
        else:
            self.scheduler = None

        self.epoch = 0

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create the optimizer for the model.

        Do not apply weight decay to batch normalization and bias parameters.
        """
        decay = set()
        no_decay = set()

        def group_parameters(module, prefix=""):
            for name, param in module.named_parameters(recurse=False):
                full_name = f"{prefix}.{name}" if prefix else name
                if not param.requires_grad:
                    continue
                if "bias" in full_name or "bn" in full_name or "batchnorm" in full_name:
                    no_decay.add(param)
                else:
                    decay.add(param)

            for child_name, child_module in module.named_children():
                group_parameters(
                    child_module,
                    prefix=f"{prefix}.{child_name}" if prefix else child_name,
                )

        group_parameters(self.model)

        optimizer_grouped_parameters = [
            {
                "params": list(decay),
                "weight_decay": self.weight_decay,
            },
            {
                "params": list(no_decay),
                "weight_decay": 0.0,
            },
        ]

        # Use a higher epsilon for numerical stability with mixed precision
        eps = 1e-7 if self.use_amp else 1e-8
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr, eps=eps)

        return optimizer

    def _train_step(self, X, Y):
        self.model.train()
        X = X.to(self.device)
        Y = Y.to(self.device)

        if self.train_transform:
            X = self.train_transform(X)

        self.optimizer.zero_grad()

        with amp.autocast(
            enabled=self.use_amp, dtype=torch.bfloat16, device_type=self.device
        ):
            outputs = self.model(X)
            loss = self.criterion(outputs, Y)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.scheduler:
            self.scheduler.step()

        return loss.item()

    def _test_step(self, X, Y):
        self.model.eval()
        X = X.to(self.device)
        Y = Y.to(self.device)

        with (
            torch.inference_mode(),
            amp.autocast(
                enabled=self.use_amp, dtype=torch.bfloat16, device_type=self.device
            ),
        ):
            outputs = self.model(X)
            loss = self.criterion(outputs, Y)

        # Update metrics
        predictions = F.sigmoid(outputs)
        for metric in self.metrics.values():
            metric.update(predictions, Y)

        return loss.item()

    def evaluate(self) -> dict:
        total_loss = 0.0

        with torch.no_grad():
            for X, Y in self.test_loader:
                loss = self._test_step(X, Y)
                total_loss += loss

        # Calculate metrics
        metrics_results = {
            "test_loss": total_loss / len(self.test_loader),
        }
        for metric_name, metric in self.metrics.items():
            metrics_results[metric_name] = metric.compute()
            metric.reset()

        return metrics_results

    def save_model(self):
        model_path = self.model_root / f"{self.model_name}.pt"
        torch.save(self.model.state_dict(), model_path)
        self.run.log_artifact(model_path, type="model", name=f"{self.model_name}")

    def train(self):
        print(f"Training {self.model_name} for {self.epochs} epochs...")
        print(f"Training on {len(self.train_loader)} batches of size {self.batch_size}")
        total_steps = self.batch_size * len(self.train_loader) * self.epochs

        with tqdm(
            total=total_steps,
            desc="Training",
            unit="img",
        ) as progress_bar:
            for epoch in range(self.epochs):
                self.epoch += 1

                train_loss = 0.0
                for X, Y in self.train_loader:
                    loss = self._train_step(X, Y)
                    train_loss += loss
                    progress_bar.update(self.batch_size)
                    progress_bar.set_postfix({"loss": f"{loss:.3f}"})

                train_loss /= len(self.train_loader)

                eval_results = self.evaluate()
                progress_bar.set_postfix({"test_loss": eval_results["test_loss"]})

                self.run.log(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "learning_rate": self.scheduler.get_last_lr()[0]
                        if self.scheduler
                        else self.lr,
                    }
                    | eval_results
                )
                self.save_model()

                print(
                    f"Epoch {epoch}/{self.epochs}, Train Loss: {train_loss:.4f}, Test Loss: {eval_results['test_loss']:.4f}"
                )

        self.run.finish()


def main():
    from dataset import GalaxyZooClassDataset
    from model import EfficientNetZooModel
    import torchvision.transforms.v2 as transforms

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    extra_train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.AutoAugment(
                policy=transforms.AutoAugmentPolicy.IMAGENET,
            ),
        ]
    )

    dataset = GalaxyZooClassDataset(
        root="./Pierre/dataset",
        transform=transform,
        n_rows=10000,
    )
    print(f"Dataset loaded with {len(dataset)} samples.")

    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])

    model = EfficientNetZooModel(output_labels=dataset.labels, dropout=0.5)

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=128,
        lr=1e-3,
        weight_decay=1e-2,
        freeze_blocks=0,
        train_transform=extra_train_transform,
        problem_type="multiclass",
        epochs=25,
        lr_scheduler="cosine_warmup",
        use_amp=True,
    )

    trainer.train()


if __name__ == "__main__":
    main()
