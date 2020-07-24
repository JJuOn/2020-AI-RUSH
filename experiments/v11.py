import torch
from torch import cuda
from torch import optim, nn
from models.resnet50 import resnet50

from torchvision import transforms

input_size=(256,256,3)
classes=['normal','monotone','screenshot','unknown']
config={
    'batch_size':64,
    'train_split':0.8,
    'lr':0.000005,
    'n_epochs':100,
    'print_every':1,
    'cuda':True if cuda.is_available() else False,
    'model':resnet50(pretrained=True,freeze=True),
    'criterion':nn.CrossEntropyLoss(),
    'transform':transforms.Compose([
            transforms.Resize(size=224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ]),
    'early_stop':100,
    'balance':True
}
config['optimizer']=optim.Adam(config['model'].parameters(), lr=config['lr'])
config['scheduler']=None