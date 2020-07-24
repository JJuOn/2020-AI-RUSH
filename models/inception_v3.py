import torchvision
import torch.nn as nn
from torchvision import models

def inception_v3(pretrained,freeze):
    model=models.inception_v3(pretrained=pretrained)
    if freeze:
        # disable aux_logits
        model.aux_logits=False
        # freeze layers
        for param in model.parameters():
            param.requires_grad=False
    n_inputs=model.fc.in_features
    model.fc=nn.Sequential(
        nn.Linear(n_inputs,1024),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(1024,4),
        nn.LogSoftmax(dim=1)
    )
    return model