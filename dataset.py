import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
from PIL import Image
from nsml import DATASET_PATH


class SpamDataset(Dataset):
    def __init__(self,transform,balance):
        self.df=pd.read_csv(os.path.join(DATASET_PATH,'train','train_label'))
        # only include labeled data
        self.df=self.df[self.df['annotation']!=-1]
        self.transform=transform
        self.data_size=[0 for i in range(4)]
        print('Before balancing:')
        for i in range(4):
            self.data_size[i]=self.df['annotation'].value_counts()[i]
            print('class: {}, data_size: {}'.format(i,self.data_size[i]))
        if balance:
            min_class=np.argmin(self.data_size)
            self.df2=pd.DataFrame(columns=['filename','annotation'])
            for i in range(4):
                self.df2=self.df2.append(self.df[self.df['annotation']==i].sample(n=self.data_size[min_class],replace=True),ignore_index=True)
            self.df=self.df2.copy()
            print('After balancing:')
            for i in range(4):
                self.data_size[i]=self.df['annotation'].value_counts()[i]
                print('class: {}, data_size: {}'.format(i,self.data_size[i]))



    def __len__(self):
        return len(self.df['filename'])

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx=idx.tolist()

        image_name=self.df.loc[idx,'filename']
        annotation=self.df.loc[idx,'annotation']
        img=Image.open(os.path.join(DATASET_PATH,'train','train_data',image_name))
        img=self.transform(img)
        return img, annotation

class TestDataset(Dataset):
    def __init__(self,transform,root_path):
        self.root_path=root_path
        self.files=os.listdir(os.path.join(self.root_path,'test_data'))
        self.transform=transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx=idx.tolist()

        image_name=self.files[idx]
        img=Image.open(os.path.join(self.root_path,'test_data',image_name))
        img=self.transform(img)
        return img, image_name