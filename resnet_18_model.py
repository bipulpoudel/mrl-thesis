# FILE: src/resnet_18_model.py (RECOMMENDED REPLACEMENT)
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# In your actual project, you would modify your src/resnet_18_model.py file
class ResNet18_CIFAR(nn.Module):
    # This is a standard ResNet18 implementation.
    # The key change is the addition of the get_embedding method.
    def __init__(self, num_classes=10, pretrained=False):
        super(ResNet18_CIFAR, self).__init__()
        if pretrained:
            self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.resnet = resnet18(weights=None) # Not using pretrained weights for CL from scratch
        
        # CIFAR-10 images are 32x32, so we modify the first conv layer
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity() # Remove max pooling for small images #type: ignore
        
        # The final fully connected layer
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self, x, get_embedding=False):
        """Forward pass through the ResNet18 model"""
        if get_embedding:
            # Return embeddings before the final classification layer
            x = self.resnet.conv1(x)
            x = self.resnet.bn1(x)
            x = self.resnet.relu(x)
            # Note: maxpool is Identity for CIFAR, so we skip it
            x = self.resnet.layer1(x)
            x = self.resnet.layer2(x)
            x = self.resnet.layer3(x)
            x = self.resnet.layer4(x)
            x = self.resnet.avgpool(x)
            x = torch.flatten(x, 1)
            return x
        else:
            return self.resnet(x)
        
    def get_embedding(self, x):
        """Get embeddings before the final classification layer"""
        return self.forward(x, get_embedding=True)
    
    def get_features(self, x):
        """Get features (embeddings) for TPG compatibility"""
        return self.get_embedding(x)
        
    # Expose the fc layer for easy access during replay
    @property
    def fc(self):
        return self.resnet.fc

    @fc.setter
    def fc(self, new_fc_layer):
        self.resnet.fc = new_fc_layer
# End of placeholder model