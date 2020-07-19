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

def train(model,criterion,optimizer,train_loader,val_loader,save_location,early_stop,n_epochs,print_every):
    val_loss_min=np.inf
    val_abnormal_f1_score_max=-np.inf
    stop_count=0
    val_max_acc=-np.inf
    history=[]
    model.epochs=0
    for epoch in range(n_epochs):
        train_loss=0
        val_loss=0
        train_acc=0
        val_acc=0
        
        model.train()
        ii=0
        for data, label in train_loader:
            ii+=1
            data, label=data.to(device),label.to(device)
            optimizer.zero_grad()
            output=model(data)
            loss=criterion(output,label)
            loss.sum().backward()
            optimizer.step()

            train_loss+=loss.item()*data.size(0)
            
            _,pred=torch.max(output,dim=1)
            correct_tensor=pred.eq(label.data.view_as(pred))
            accuracy=torch.mean(correct_tensor.type(torch.FloatTensor))
            train_acc+=accuracy.item()*data.size(0)

            if ii%10==0:
                print('Epoch: {}\t{:.2f}% complete.'.format(epoch,100*(ii+1)/len(train_loader)))
        model.epochs+=1
        with torch.no_grad():
            model.eval()
            preds=[]
            trues=[]
            for data, label in val_loader:
                data, label=data.to(device), label.to(device)

                output=model(data)
                loss=criterion(output,label)
                val_loss+=loss.item()*data.size(0)
                _,pred=torch.max(output,dim=1)
                correct_tensor=pred.eq(label.data.view_as(pred))
                accuracy=torch.mean(correct_tensor.type(torch.FloatTensor))
                val_acc+=accuracy.item()*data.size(0)
                preds.append(pred.detach().cpu())
                trues.append(label.detach().cpu())
            preds=np.concatenate(preds)
            trues=np.concatenate(trues)
            cr=classification_report(trues,preds,labels=[0,1,2,3],target_names=['normal','monotone','screenshot','unknown'],output_dict=True,zero_division=0)
            val_abnormal_f1_score=(cr['monotone']['f1-score']*cr['screenshot']['f1-score']*cr['unknown']['f1-score'])**(1/3)
            train_loss=train_loss/len(train_loader.dataset)
            val_loss=val_loss/len(val_loader.dataset)

            train_acc=train_acc/len(train_loader.dataset)
            val_acc=val_acc/len(val_loader.dataset)

            history.append([train_loss,val_loss,train_acc,val_acc])

            if (epoch+1)%print_every==0:
                print('\nEpoch: {}\tTraining Loss: {:.4f}\tValidation Loss: {:.4f}'.format(epoch,train_loss,val_loss))
                print('\t\tTraining Accuracy: {:.2f}%\tValidation Accuracy: {:.2f}%'.format(100*train_acc,100*val_acc))
                print('\t\tClassification Report:{}'.format(cr))
                print('\t\tF1 Score for Abnormal Class: {}'.format(val_abnormal_f1_score))

            if val_abnormal_f1_score>val_abnormal_f1_score_max:
                nsml.save(str(epoch))
                stop_count=0
                val_loss_min=val_loss
                val_max_acc=val_acc
                val_abnormal_f1_score_max=val_abnormal_f1_score
                best_epoch=epoch
            else:
                stop_count+=1
                if stop_count>=early_stop:
                    print('\nEarly Stopping Total epochs: {}. Best epoch: {} with loss: {:.2f} and ac: {:.2f}% F1 Score for Abnormal Class: {}'.format(epoch,best_epoch,val_loss_min,100*val_acc,val_abnormal_f1_score_max))
                    return
    model.optimizer=optimizer
    print('Best epoch: {} with loss: {:.2f} and ac: {:.2f}% f1-score: {}'.format(best_epoch,val_loss_min,100*val_acc,val_abnormal_f1_score_max))
    return

def evaluate(model,root_path):
    model.eval()
    test_set=TestDataset(config['transform'],root_path)
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
    model=nn.DataParallel(model)
    global device
    device=torch.device('cuda:0' if config['cuda'] else 'cpu')
    bind_nsml(model,criterion)
    if parsed_args.pause:
        nsml.paused(scope=locals())
    if parsed_args.mode=='train':
        model=model.to(device)
        criterion=criterion.to(device)
        dataset=SpamDataset(transform=config['transform'])
        total_len=len(dataset)
        # sampler=WeightedRandomSampler(dataset.data_weight,config['batch_size'],replacement=True)
        train_set, val_set=torch.utils.data.random_split(dataset,[int(config['train_split']*total_len),total_len-int(config['train_split']*total_len)])
        train_loader=torch.utils.data.DataLoader(train_set,batch_size=config['batch_size'],shuffle=True,num_workers=2)#,sampler=sampler)
        val_loader=torch.utils.data.DataLoader(val_set,batch_size=config['batch_size'],shuffle=True,num_workers=2)
        train(model,criterion,optimizer,train_loader,val_loader,None,config['early_stop'],config['n_epochs'],1)


    