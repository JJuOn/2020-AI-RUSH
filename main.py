import os
from importlib import import_module
import argparse
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data.sampler import WeightedRandomSampler
from torch import cuda

import torchvision

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report

import nsml

from dataset import SpamDataset, TestDataset
from train import train, mixed_train

def evaluate(model,root_path):
    model.eval()
    test_set=TestDataset(transform.Compose([
            transforms.Resize(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ]),root_path)
    test_loader=torch.utils.data.DataLoader(test_set,batch_size=config['batch_size'],shuffle=False)
    filenames=[]
    y_preds=[]

    model=model.to(device)
    with torch.no_grad():
        for data, filenames_tensor in test_loader:
            if config['cuda']:
                data = data.to(device)
            output=model(data)
            _,pred=torch.max(output,dim=1)
            filenames.append(filenames_tensor)
            y_preds.append(pred.tolist())
    filenames=np.concatenate(filenames,0)
    y_preds=np.concatenate(y_preds,0)
    df=pd.DataFrame({'filename':filenames,'y_pred':y_preds})
    return df


def bind_nsml(model,criterion):
    def load(dir_name):
        state=torch.load(os.path.join(dir_name,'model.pth'))
        model.load_state_dict(state['model'])
        criterion.load_state_dict(state['criterion'])
        print('Loaded')

    def save(dir_name):
        state={
            'model':model.state_dict(),
            'criterion':criterion.state_dict()
        }
        torch.save(state,os.path.join(dir_name,'model.pth'))
        print('Saved')

    def infer(root_path):
        return evaluate(model,root_path)

    nsml.bind(save=save,load=load,infer=infer)


if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument('--experiment_name',type=str,default='v1')
    args.add_argument('--pause',type=int,default=0)
    args.add_argument('--mode',type=str,default='train')
    parsed_args=args.parse_args()
    global config
    config=import_module('experiments.{}'.format(parsed_args.experiment_name)).config

    model=config['model']
    criterion=config['criterion']
    optimizer=config['optimizer']
    scheduler=config['scheduler']

    model=nn.DataParallel(model)
    global device
    device=torch.device('cuda:0' if config['cuda'] else 'cpu')
    bind_nsml(model,criterion)

    if parsed_args.pause:
        nsml.paused(scope=locals())
    if parsed_args.mode=='train':
        print('mode: train')
        model=model.to(device)
        criterion=criterion.to(device)
        dataset=SpamDataset(transform=config['transform'],balance=config['balance'])
        total_len=len(dataset)
        train_set, val_set=torch.utils.data.random_split(dataset,[int(config['train_split']*total_len),total_len-int(config['train_split']*total_len)])
        train_loader=torch.utils.data.DataLoader(train_set,batch_size=config['batch_size'],shuffle=True,num_workers=2)
        val_loader=torch.utils.data.DataLoader(val_set,batch_size=config['batch_size'],shuffle=True,num_workers=2)
        train(model,criterion,optimizer,scheduler,train_loader,val_loader,config['early_stop'],config['n_epochs'],1,device)
    elif parsed_args.mode=='mixed_train':
        print('mode: mixed train')
        model=model.to(device)
        criterion=criterion.to(device)
        dataset=SpamDataset(transform=config['transform'],balance=config['balance'])
        total_len=len(dataset)
        train_set, val_set=torch.utils.data.random_split(dataset,[int(config['train_split']*total_len),total_len-int(config['train_split']*total_len)])
        train_loader=torch.utils.data.DataLoader(train_set,batch_size=config['batch_size'],shuffle=True,num_workers=2)
        val_loader=torch.utils.data.DataLoader(val_set,batch_size=config['batch_size'],shuffle=True,num_workers=2)
        mixed_train(model,criterion,optimizer,scheduler,train_loader,val_loader,device)


    