import torch 
import numpy as np
from utils.DataPre import get_features,get_regions
from torch.utils.data import Dataset
from PIL import Image



class BrainDataset(Dataset):
    def __init__(self,data,labels,device,imdim=(400,400),transform=None) -> None:
        self.data = data 
        self.labels = labels 
        self.transform = transform
        self.device = device 
        self.imdim = imdim

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        
        image_file = self.data[idx]
        image = np.array(Image.open(image_file).convert("L").resize((self.imdim)))
        if self.transform :
            image = self.transform(image)
        
        
        image = torch.tensor(image,dtype=torch.float32)
        label = torch.tensor(self.labels[idx],dtype = torch.float32)
        
        return image,label

class Transform:
    def __init__(self, rows, columns, num_bins):
        self.rows = rows
        self.columns = columns
        self.num_bins = num_bins

    def __call__(self, image):
        regions = get_regions(image, self.rows, self.columns)
        image = get_features(regions, self.num_bins)
        return image














