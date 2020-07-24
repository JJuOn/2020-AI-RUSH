import torchvision
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

def efficientnet_b6(pretrained,freeze):
    if pretrained:
        model=EfficientNet.from_pretrained("efficientnet-b6",num_classes=4)
    else:
        print('Only pretrained is avaliable for EfficientNet-b6')
        exit(1)
    # freeze layers
    if freeze:
        for n,p in model.named_parameters():
            if '_fc' not in n:
                p.requires_grad=False

    n_inputs=model._fc.in_features
    model._fc=nn.Linear(n_inputs,4)
    return model

def efficientnet_b3(pretrained,freeze):
    if pretrained:
        model=EfficientNet.from_pretrained("efficientnet-b3",num_classes=4)
    else:
        print('Only pretrained is avaliable for EfficientNet-b6')
        exit(1)
    # freeze layers
    if freeze:
        for n,p in model.named_parameters():
            if '_fc' not in n:
                p.requires_grad=False

    n_inputs=model._fc.in_features
    model._fc=nn.Linear(n_inputs,4)
    return model