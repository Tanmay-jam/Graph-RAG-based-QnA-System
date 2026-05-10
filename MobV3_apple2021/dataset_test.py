from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import torch
from PIL import Image
import numpy as np
import os
from glob import glob



def read_data(batchsize, data_dir = '/DATA/ishwar_2221cs30/Work_2/Apple2021/data', num_workers=8):

    # data transform dictionary
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation((-5,5)),
            transforms.RandomResizedCrop((224)),     #(384,512)
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224,224)),   #(480,640)
            transforms.CenterCrop((224)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # image dataset dictionary
    image_datasets = {x: dataset(mode=x, transform=data_transforms[x])
                      for x in ['train', 'val']}
    
    data_loader_train = DataLoader(dataset=image_datasets['train'],
                                   batch_size=batchsize,
                                   shuffle=True,
                                   num_workers=8,
                                   pin_memory=True
                                   )
    data_loader_test = DataLoader(dataset=image_datasets['val'],
                                  batch_size=batchsize,
                                  shuffle=False,
                                  num_workers=8,
                                  pin_memory=True
                                  )
    
    return data_loader_train,data_loader_test

class dataset(Dataset):

    def __init__(self, data_dir= '/DATA/ishwar_2221cs30/Work_2/Apple2021/data', mode="train", transform=None):

        self.root = data_dir
        self.mode = mode
        self.T = transform
        self.labels = ["complex", "healthy", "mildew", "rust", "scab", "spot"]    # change according to classes 
        self.labelsdict = {"complex": 0, "healthy": 1, "mildew": 2, "rust": 3, "scab": 4, "spot": 5} # change according to classes
        self.idlist = []
        for i in range(len(self.labels)):
            self.idlist.extend(glob(os.path.join(self.root, self.mode, self.labels[i], "*.jpg")))
        
    def __len__(self):
        return len(self.idlist)

    def __getitem__(self, idx):

        # get path
        imgpath = self.idlist[idx]
        id = imgpath.split("/")[-1].split(".jpg")[0]
        

        # extract image
        with open(imgpath, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")


        # extract label
        label = self.labelsdict[imgpath.split("/")[-2]]
        
        # transform
        state = torch.get_rng_state()
        img = self.T(img)
        img = transforms.functional.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        return img, label, imgpath
        
        

