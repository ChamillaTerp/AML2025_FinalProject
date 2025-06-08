
import torch
import torch.nn as nn
import pytorch_lightning as L

class LitGalaxyModel(L.LightningModule):
    def __init__(self, model, criterion, learning_rate = 0.001):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.save_hyperparameters(ignore = ['model', 'criterion'])

        # Add lists to store loss values
        self.train_losses = []
        self.val_losses = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        self.train_losses.append(loss.item())  # <-- Save the loss
        self.log('train_loss', loss, prog_bar = True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        self.val_losses.append(loss.item())  # <-- Save the loss
        self.log('val_loss', loss, prog_bar = True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr = self.hparams.learning_rate)