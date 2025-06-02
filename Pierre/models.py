import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math

from typing import Sequence


class EfficientNetZooModel(nn.Module):
    def __init__(self, output_labels: Sequence, dropout: float = 0.2):
        super(EfficientNetZooModel, self).__init__()

        self.output_names = output_labels
        self.output_dim = len(output_labels)

        self.features = torchvision.models.efficientnet_v2_m().features

        for param in self.features.parameters():
            param.requires_grad = True

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 10))

        for m in self.classifier.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
