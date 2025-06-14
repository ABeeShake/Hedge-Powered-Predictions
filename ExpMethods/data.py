import numpy as np
import pandas as pd
import torch
import lightning as L

from ExpMethods.utils import *

def transform_minute_data(df:pd.DataFrame, **kwargs):
    
    n_days = kwargs.get("n_days", 0)
    return_type = kwargs.get("return_type",torch.Tensor)
    device = kwargs.get("device","cpu")
    
    data = df.copy()
    
    data["Race"] = data["Race"].map(
        {"African American Black": 1,
         "African American": 2,
         "Hispanic/Latino": 3,
         "White": 4})
    
    data["Timestamp"] = pd.to_datetime(data["Timestamp"])
    
    if n_days > 0:
        days = data.Timestamp.dt.day.unique()
        data = data.loc[data.Timestamp.dt.day.isin(days[n_days-1:n_days]),:]
    
    data.drop(["Unnamed: 0","Timestamp"], axis = 1, inplace = True)
    
    target_col = "Libre.GL" if "Libre.GL" in data.columns else "Dexcom.GL"
    
    data.insert(data.shape[1]-1, target_col, data.pop(target_col))
    
    if return_type == torch.Tensor:
    
        return torch.tensor(data.to_numpy()).to(device).to(torch.float32)
    else:
        return data
    

class Sampler():
    
    def __init__(self, N, b, h, **kwargs):
        
        h_first = kwargs.get("h_first", True)
        
        self.N = N
        
        if h_first:
            self.h = min(h, N - 1)
            self.b = min(b, N - h)
        else:
            self.b = min(b, N - 1)
            self.h = min(h, N - b)
        
    def __iter__(self):
        
        idx = np.arange(self.N - self.h)
        
        batches = []
        
        for _ in range(self.b):
            
            batches.append(
                np.random.choice(
                    idx, 
                    self.b, 
                    replace = False
                    ).tolist()
                )
        return iter(batches)
        
    
class DiabetesMinuteDataset(torch.utils.data.Dataset):
    
    def __init__(self, data: pd.DataFrame, horizon = 1, transform = None):

        self.df = data
        self.transform = transform
        self.h = horizon
    
    def __len__(self):
        
        return len(self.df)
    
    def __getitem__(self, idx):
            
        inputs = self.df.iloc[idx:idx+self.h,:].to_numpy("float32")
        #input shape = (len(idx), time_length, n_features)
        targets = self.df.iloc[idx + 1:idx + self.h + 1, -1].to_numpy().astype("float32")
        
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
        self.test_h = kwargs.get("max_horizon", 1)
        self.h_first = kwargs.get("h_first",True)
        self.num_workers = kwargs.get("num_workers",511)
        
        if self.h_first:
            self.train_h = min(self.test_h, len(self.train) - 1)
            self.b = min(self.b, len(self.train) - self.train_h)
        else:
            self.b = min(self.b, len(self.train) - 1)
            self.train_h = min(self.test_h, len(self.train) - self.b)
    
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
        
        sampler_params = dict(
            N = len(self.train_dataset),
            h = self.train_h,
            b = self.b,
            h_first = self.h_first
            )
        
        return (
        torch.utils.data.DataLoader(
            self.train_dataset,
            batch_sampler = Sampler(**sampler_params),
            num_workers = self.num_workers)
            )
    def test_dataloader(self):
        
        sampler_params = dict(
            N = len(self.test_dataset),
            h = self.test_h,
            b = self.b,
            h_first = self.h_first
            )
        
        return (
        torch.utils.data.DataLoader(
            self.test_dataset,
            batch_sampler = Sampler(**sampler_params),
            num_workers = self.num_workers)
            )
