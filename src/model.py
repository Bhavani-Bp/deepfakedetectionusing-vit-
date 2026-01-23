import torch
import torch.nn as nn
from torchvision import models

class DeepfakeDetector(nn.Module):
    def __init__(self, pretrained=True):
        """
        Deepfake Detection Model based on EfficientNet-B0.
        
        Args:
            pretrained (bool): Whether to load ImageNet weights.
        """
        super(DeepfakeDetector, self).__init__()
        
        # Load EfficientNet-B0
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.model = models.efficientnet_b0(weights=weights)
        
        # Modify the classifier head
        # EfficientNet-B0 classifier is a Sequential with Dropout and Linear
        # The final Linear layer matches num_classes (1000 for ImageNet)
        # We want 1 output (binary classification logits)
        
        in_features = self.model.classifier[1].in_features
        
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features, 1) 
        )
        
        print("Initialized DeepfakeDetector (EfficientNet-B0)")

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (tensor): Input images of shape (B, 3, H, W)
            
        Returns:
            tensor: Logits of shape (B, 1)
        """
        return self.model(x)
