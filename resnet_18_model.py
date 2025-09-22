# FILE: src/resnet_18_model.py
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18_CIFAR(nn.Module):
    """ResNet-18 adapted for CIFAR-size inputs (32×32) with optional embeddings."""
    def __init__(self, num_classes=10, pretrained=False):
        super().__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        # CIFAR stem: 3×3 conv, stride 1; remove initial maxpool
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()

        in_feats = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_feats, num_classes)
        self.feature_dim = in_feats  # handy for heads

    def forward(self, x, get_embedding: bool = False):
        if get_embedding:
            m = self.resnet
            x = m.conv1(x); x = m.bn1(x); x = m.relu(x)
            x = m.layer1(x); x = m.layer2(x); x = m.layer3(x); x = m.layer4(x)
            x = m.avgpool(x)
            return torch.flatten(x, 1)
        return self.resnet(x)

    def get_embedding(self, x):
        return self.forward(x, get_embedding=True)