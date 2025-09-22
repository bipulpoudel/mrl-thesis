# FILE: src/mrl_resnet_18_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class MRL_ResNet18_CIFAR(nn.Module):
    def __init__(self, num_classes=10, pretrained=False, mrl_dims=None, mode='multi_head_explicit'):
        """
        Initializes the MRL ResNet-18 model for CIFAR.

        Args:
            num_classes (int): The number of output classes.
            pretrained (bool): Whether to use ImageNet pre-trained weights.
            mrl_dims (list of int): A list of feature dimensions for the nested heads.
            mode (str): The MRL implementation mode.
                        'single_head_efficient': Uses a single, full-size linear layer and
                                                 slices features/weights on the fly. Most efficient.
                        'multi_head_explicit': Creates a separate nn.Linear layer for each
                                               MRL dimension. Less efficient, used for validation.
        """
        super().__init__()
        
        if mrl_dims is None or not mrl_dims:
            raise ValueError("mrl_dims must be a list of feature dimensions, e.g., [128, 256, 512]")
        
        # Ensure dimensions are sorted, which is crucial for nested representations
        self.mrl_dims = sorted(mrl_dims)
        self.mode = mode
        
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        # Adapt for CIFAR-sized inputs (32x32)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()

        # Get the feature dimension from the ResNet backbone (it's 512)
        in_feats = self.resnet.fc.in_features

        if self.mode == 'single_head_efficient':
            # --- Original Efficient Implementation ---
            # Replace the final layer with one that outputs based on the largest MRL dimension.
            # All smaller heads will be 'virtual' and created by slicing this one.
            if self.mrl_dims[-1] != in_feats:
                print(
                    f"Warning (Single-Head Mode): The largest MRL dimension ({self.mrl_dims[-1]}) does not match "
                    f"the model's natural feature dimension ({in_feats}). Make sure this is intentional."
                )
            self.resnet.fc = nn.Linear(in_feats, num_classes)
        
        elif self.mode == 'multi_head_explicit':
            # --- New Explicit Multi-Head Implementation ---
            # Remove the original fc layer entirely. We'll replace it with a module list.
            self.resnet.fc = nn.Identity()
            
            # Create a separate linear layer for each MRL dimension.
            self.mrl_heads = nn.ModuleList()
            for dim in self.mrl_dims:
                self.mrl_heads.append(nn.Linear(dim, num_classes))
        
        else:
            raise ValueError(f"Invalid mode '{self.mode}'. Choose from 'single_head_efficient' or 'multi_head_explicit'.")

    def get_embedding(self, x):
        m = self.resnet
        x = m.conv1(x); x = m.bn1(x); x = m.relu(x)
        x = m.layer1(x); x = m.layer2(x); x = m.layer3(x); x = m.layer4(x)
        x = m.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        # 1. Get the final feature embedding from the ResNet body
        features = self.get_embedding(x) # Shape: (batch_size, 512)

        outputs = []
        if self.mode == 'single_head_efficient':
            # --- Original Efficient Forward Pass ---
            for dim in self.mrl_dims:
                sub_features = features[:, :dim]
                sub_weights = self.resnet.fc.weight[:, :dim]
                sub_logits = F.linear(sub_features, sub_weights, self.resnet.fc.bias)
                outputs.append(sub_logits)
        
        elif self.mode == 'multi_head_explicit':
            # --- New Explicit Multi-Head Forward Pass ---
            # We iterate through the dimensions and our explicit heads simultaneously
            for i, dim in enumerate(self.mrl_dims):
                # Slice the features to maintain the nested property
                sub_features = features[:, :dim]
                
                # Apply the corresponding dedicated linear layer
                head = self.mrl_heads[i]
                sub_logits = head(sub_features)
                outputs.append(sub_logits)
        
        return outputs