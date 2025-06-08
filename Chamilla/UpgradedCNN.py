import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class UpgradedCNN(nn.Module):
    def __init__(self, num_outputs = 8):
        super(UpgradedCNN, self).__init__()                                        # Initialize the parent class                                            

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(16),                                                   # Batch normalization to stabilize learning and improve convergence                
            nn.ReLU(inplace = True),                                              # Use inplace ReLU to save memory                 
            nn.MaxPool2d(kernel_size = 2),                                        # Max pooling layer to reduce spatial dimensions     

            nn.Conv2d(16, 32, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2),

            nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2),

            nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))                             # Output size: [batch, 128, 1, 1]

        self.classifier = nn.Sequential(
            nn.Flatten(),                                                           # [batch, 128]
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(256, num_outputs)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)

        return x
