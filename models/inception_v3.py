import torchvision
import torch.nn as nn
from torchvision import models

inception_v3=models.inception_v3(pretrained=True)

# disable aux_logits
inception_v3.aux_logits=False

# freeze layers
for param in inception_v3.parameters():
    param.requires_grad=False

n_inputs=inception_v3.fc.in_features
inception_v3.fc=nn.Sequential(
    nn.Linear(n_inputs,1024),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(1024,4),
    nn.LogSoftmax(dim=1)
)