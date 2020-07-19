import torch
from torch import cuda
from torch import optim, nn
from models.inception_v3 import inception_v3

from torchvision import transforms

input_size=(256,256,3)
classes=['normal','monotone','screenshot','unknown']
config={
    'batch_size':128,
    'train_split':0.8,
    'lr':0.000005,
    'n_epochs':100,
    'print_every':1,
    'cuda':True if cuda.is_available() else False,
    'model':inception_v3,
    'criterion':nn.NLLLoss(),
    'transform':transforms.Compose([
            transforms.Resize(size=299),
            # transforms.RandomRotation(degrees=15),
            # transforms.ColorJitter(),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomResizedCrop(size=299),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ]),
    'early_stop':100
}
config['optimizer']=optim.SGD(config['model'].parameters(), lr=config['lr'])