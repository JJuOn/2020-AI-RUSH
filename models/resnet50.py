import torchvision
import torch.nn as nn
from torchvision import models

def resnet50(pretrained,freeze):
    model=models.resnet50(pretrained=True)
    if freeze:
        # freeze layers
        for param in model.parameters():
            param.requires_grad=False
    n_inputs=model.fc.in_features
    model.fc=nn.Linear(n_inputs,4)
    return model