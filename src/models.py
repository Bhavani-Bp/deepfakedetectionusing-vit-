import torch
import torch.nn as nn
import timm
import logging
from torchvision import models

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepfakeEfficientNet(nn.Module):
    """
    EfficientNet-B0 for Deepfake Detection using timm.
    Matches the naming convention (blocks.x.x) found in models/deepfake_efficientnet.pth.
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(DeepfakeEfficientNet, self).__init__()
        try:
            # Create the EfficientNet model using timm
            self.model = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=num_classes)
            logger.info("Successfully initialized DeepfakeEfficientNet (timm efficientnet_b0)")
        except Exception as e:
            logger.error(f"Error initializing EfficientNet via timm: {e}")
            raise e

    def forward(self, x):
        return self.model(x)

    def extract_features(self, x):
        """
        Extract features using timm's forward_features.
        """
        return self.model.forward_features(x)

class DeepfakeViT(nn.Module):
    """
    Vision Transformer for Single Image Deepfake Detection.
    Uses a pre-trained ViT model from timm.
    """
    def __init__(self, model_name='vit_base_patch16_224', num_classes=2, pretrained=True):
        super(DeepfakeViT, self).__init__()
        self.model_name = model_name
        try:
            # Create the ViT model
            self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
            logger.info(f"Successfully initialized {model_name}")
        except Exception as e:
            logger.error(f"Error initializing {model_name}: {e}")
            raise e

    def forward(self, x):
        """
        Forward pass for image batch.
        x: [Batch, Channels, Height, Width]
        """
        return self.vit(x)
    
    def extract_features(self, x):
        """
        Extract features for video frame sequences.
        Returns: [Batch, Features]
        """
        return self.vit.forward_features(x)

class TemporalTransformer(nn.Module):
    """
    Temporal Transformer for Video Deepfake Detection.
    Takes a sequence of frame embeddings and predicts real/fake for the video.
    """
    def __init__(self, input_dim=768, num_classes=2, num_layers=4, nhead=8, dim_feedforward=2048, dropout=0.1):
        super(TemporalTransformer, self).__init__()
        
        self.input_dim = input_dim
        
        # Encoder layer: Self-Attention + FeedForward
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True # Expected input: [Batch, Seq_Len, Features]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

        logger.info("Successfully initialized TemporalTransformer")

    def forward(self, x):
        """
        x: [Batch, Seq_Len, Features] - Features extracted from frames (e.g., by ViT)
        """
        # Pass through Transformer
        # Output: [Batch, Seq_Len, Features]
        transformed = self.transformer_encoder(x)
        
        # Global Average Pooling across the time dimension (Seq_Len)
        # We want one vector per video
        pooled = transformed.mean(dim=1) # [Batch, Features]
        
        # Classify
        logits = self.classifier(pooled)
        return logits

class PaidDeepfakeSystem(nn.Module):
    """
    Combined System Wrapper (Optional, for easy loading).
    """
    def __init__(self):
        super(PaidDeepfakeSystem, self).__init__()
        # We might keep them separate in practice to save memory, 
        # loading Image model only for images and Video model only for videos.
        pass
