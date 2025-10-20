# src/model.py
import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=2, pretrained=True, device="cpu"):
    model = models.resnet18(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model = model.to(device)
    return model
