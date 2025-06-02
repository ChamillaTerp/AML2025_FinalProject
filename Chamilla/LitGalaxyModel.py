
import torch
import torch.nn as nn
import pytorch_lightning as L

# Wrapping the model in a PyTorch Lightning module
class LitGalaxyModel(L.LightningModule):
    def __init__(self, model, criterion, learning_rate = 0.001):
        super().__init__()
        self.model = model                                                  # The CNN model                   
        self.criterion = criterion                                          # The loss function               
        self.save_hyperparameters(ignore = ['model', 'criterion'])          # Save hyperparameters except the model and criterion itself

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, prog_bar = True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss, prog_bar = True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr = self.hparams.learning_rate)