import numpy as np
import torch
import torch.optim.lr_scheduler
from sklearn.metrics import classification_report
import nsml
import torchvision.models

def train(model,criterion,optimizer,scheduler,train_loader,val_loader,early_stop,n_epochs,print_every,device):
    val_loss_min=np.inf
    val_abnormal_f1_score_max=-np.inf
    stop_count=0
    val_max_acc=-np.inf
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
            if isinstance(scheduler,torch.optim.lr_scheduler.CosineAnnealingLR):
                scheduler.step()
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

            nsml.report(summary=True,scope=locals(),step=epoch,normal=cr['normal']['f1-score'],monotone=cr['monotone']['f1-score'],screenshot=cr['screenshot']['f1-score'],unknown=cr['unknown']['f1-score'],abnormal=val_abnormal_f1_score)
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
            
            if isinstance(scheduler,torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
    model.optimizer=optimizer
    print('Best epoch: {} with loss: {:.2f} and ac: {:.2f}% f1-score: {}'.format(best_epoch,val_loss_min,100*val_acc,val_abnormal_f1_score_max))
    return

def mixed_train(model,criterion,optimizer,scheduler,train_loader,val_loader,device):
    print('Train: Fine Tune')
    model.train()
    # Fine Tuning Step - Train
    train_loss=0
    val_loss=0
    train_acc=0
    val_acc=0
    i=0
    epoch=0
    for data, label in train_loader:
        i+=1
        data,label=data.to(device),label.to(device)
        optimizer.zero_grad()
        output=model(data)
        loss=criterion(output,label)
        loss.sum().backward()
        optimizer.step()
        if isinstance(scheduler,torch.optim.lr_scheduler.CosineAnnealingLR):
            scheduler.step()
        train_loss+=loss.item()*data.size(0)
        _,pred=torch.max(output,dim=1)
        correct_tensor=pred.eq(label.data.view_as(pred))
        accuracy=torch.mean(correct_tensor.type(torch.FloatTensor))
        train_acc+=accuracy.item()*data.size(0)
        if i%10==0:
            print('Epoch: {}\t{:.2f}% complete.'.format(epoch,100*(i+1)/len(train_loader)))
    # Fine Tuning Step - Validation
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

        nsml.report(summary=True,scope=locals(),step=epoch,normal=cr['normal']['f1-score'],monotone=cr['monotone']['f1-score'],screenshot=cr['screenshot']['f1-score'],unknown=cr['unknown']['f1-score'],abnormal=val_abnormal_f1_score)
            
        print('\nEpoch: {}\tTraining Loss: {:.4f}\tValidation Loss: {:.4f}'.format(epoch,train_loss,val_loss))
        print('\t\tTraining Accuracy: {:.2f}%\tValidation Accuracy: {:.2f}%'.format(100*train_acc,100*val_acc))
        print('\t\tClassification Report:{}'.format(cr))
        print('\t\tF1 Score for Abnormal Class: {}'.format(val_abnormal_f1_score))
        nsml.save(str('finetune'))

    print('Train: Full Tune')
    # Full Tuning Step - train
    for p in model.parameters():
        required_grad=True
    if isinstance(model,torchvision.models.Inception3):
        model.aux_logits=True
    model.train()
    train_loss=0
    val_loss=0
    train_acc=0
    val_acc=0
    i=0
    epoch=1
    for data, label in train_loader:
        i+=1
        data,label=data.to(device),label.to(device)
        optimizer.zero_grad()
        output=model(data)
        loss=criterion(output,label)
        loss.sum().backward()
        optimizer.step()
        if isinstance(scheduler,torch.optim.lr_scheduler.CosineAnnealingLR):
            scheduler.step()
        train_loss+=loss.item()*data.size(0)
        _,pred=torch.max(output,dim=1)
        correct_tensor=pred.eq(label.data.view_as(pred))
        accuracy=torch.mean(correct_tensor.type(torch.FloatTensor))
        train_acc+=accuracy.item()*data.size(0)
        if i%10==0:
            print('Epoch: {}\t{:.2f}% complete.'.format(epoch,100*(i+1)/len(train_loader)))

    # Full Tuning Step - Validation
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

        nsml.report(summary=True,scope=locals(),step=epoch,normal=cr['normal']['f1-score'],monotone=cr['monotone']['f1-score'],screenshot=cr['screenshot']['f1-score'],unknown=cr['unknown']['f1-score'],abnormal=val_abnormal_f1_score)
            
        print('\nEpoch: {}\tTraining Loss: {:.4f}\tValidation Loss: {:.4f}'.format(epoch,train_loss,val_loss))
        print('\t\tTraining Accuracy: {:.2f}%\tValidation Accuracy: {:.2f}%'.format(100*train_acc,100*val_acc))
        print('\t\tClassification Report:{}'.format(cr))
        print('\t\tF1 Score for Abnormal Class: {}'.format(val_abnormal_f1_score))
        nsml.save(str('fulltune'))
    return