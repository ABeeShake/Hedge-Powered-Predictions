import numpy as np
import pandas as pd
import torch
from typing import *
from ExpMethods.utils import *

def transform_minute_data(data:pd.DataFrame, **kwargs):
    
    n_days = kwargs.get("n_days", 1)
    return_type = kwarg_parse(kwargs,"return_type",torch.Tensor)
    device = kwargs.get("device","cpu")
    
    data["Race"] = data["Race"].map(
        {"African American Black": 1,
         "African American": 2,
         "Hispanic/Latino": 3,
         "White": 4})

    data["Timestamp"] = pd.to_datetime(data["Timestamp"])

    days = data.Timestamp.dt.day.unique()
    
    data = data[data.Timestamp.dt.day.isin(days[n_days-1:n_days])]
    
    data.drop("Timestamp", axis = 1, inplace = True)
    
    target_col = "Libre.GL" if "Libre.GL" in data.columns else "Dexcom.GL"
    
    data.insert(data.shape[1]-1, target_col, data.pop(target_col))
    
    if return_type == torch.Tensor:
    
        return torch.tensor(data.to_numpy()).to(device).to(torch.float32)
    else:
        return data
    
    
class DiabetesMinuteDataset(torch.utils.data.Dataset):
    
    def __init__(self, data: pd.DataFrame, horizon = 1, transform = None):

        self.df = data
        self.transform = transform
        self.h = horizon
    
    def __len__(self):
        
        return len(self.df)
    
    def __getitem__(self, idx):
        
        idx = min(idx, self.__len__()-self.h-1)
            
        inputs = self.df.iloc[idx:idx+self.h,:].to_numpy("float32")
        #input shape = (len(idx), time_length, n_features)
        targets = self.df.iloc[idx + self.h, -1].astype("float32")
        
        sample = (inputs, targets)
        
        if self.transform:
            self.transform(sample)
            
        return sample


class MinuteDataLightningDataModule(L.LightningDataModule):
    
    def __init__(self, x_train, x_test, transform = None, **kwargs):
        
        super().__init__()
        self.train = x_train
        self.test = x_test
        self.transform = transform
        self.b = kwargs.get("batch_size", 10)
        self.train_h = kwargs.get("train_horizon", 1)
        self.test_h = kwargs.get("test_horizon", 1)
    
    def prepare_data(self):
        pass
        
    def setup(self, stage = None):
        
        if stage == "fit" or stage is None:
            
            self.train_dataset = DiabetesMinuteDataset(
                self.train, 
                horizon = self.train_h, 
                transform = self.transform
                )
                
        if stage == "test" or stage is None:
            
            self.test_dataset = DiabetesMinuteDataset(
                self.test, 
                horizon = self.test_h, 
                transform = self.transform
                )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
        self.train_dataset, 
        batch_size = self.b,
        shuffle = False,
        num_workers = 511)
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
        self.test_dataset,
        batch_size = self.b,
        shuffle = False,
        num_workers = 511)
