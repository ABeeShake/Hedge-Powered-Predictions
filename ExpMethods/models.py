import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchdyn.core import NeuralODE
from torchdyn.nn import DataControl, DepthCat
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from ExpMethods.utils import *


class NODE(nn.Module):
    
    def __init__(self, f, device, **kwargs):
        
        super().__init__()
        self.device = device
        self.model = NeuralODE(f, **kwargs).to(device)
        
    def configure_optimizer(self, **kwargs):

        lr = kwarg_parse(kwargs, "lr",1e-3)
        weight_decay = kwarg_parse(kwargs, "weight_decay",1e-1)
        
        return optim.Adam(self.model.parameters(), lr = lr, weight_decay = weight_decay)
    
    def train_step(self, x_train, **kwargs):
        
        b = kwarg_parse(kwargs, "batch_size", 10)
        h = kwarg_parse(kwargs, "horizon", 1)
        
        x,y = self._get_batch(x_train, batch_size = b, horizon = h)
        t_span = torch.linspace(0,h,h+1)
            
        y_hat = self.model(x, t_span)[1][-1,:,-1]
        
        # print(f"y_hat shape: {y_hat.shape}")
            
        l = nn.functional.mse_loss(y,y_hat)
        
        return l
    
    def predict(self, x, horizon = 1):
        
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        t_span = torch.linspace(0,horizon,horizon+1)
        y_hat = self.model(x,t_span)[1][-1,:,-1]
        
        return y_hat
    
    def _get_batch(self,x_train,**kwargs):
        
        batch_size = kwarg_parse(kwargs, "batch_size", 10)
        
        m = len(x_train)
        b = min(batch_size, m-1)
        
        #tradeoff between horzion and batch size:
        #choose maximum horizon that provides batch size >= 10
        
        h = kwarg_parse(kwargs,"horizon",(m-b))
        
        h = min(h, m - b)
        
        xs = x_train[:-h]
        ys = x_train[h:,-1]
        
        s = torch.from_numpy(
            np.random.choice(
                np.arange(
                    m-h, 
                    dtype = np.int64
                    ),
                    b,
                    replace = False
                    ))
        
        return xs, ys


class LSTM(nn.Module):
    
    def __init__(self, f, device, **kwargs):
        
        super().__init__()
        self.device = device
        self.model = NeuralODE(f, **kwargs).to(device)
        
    def configure_optimizer(self, **kwargs):

        lr = kwarg_parse(kwargs, "lr",1e-3)
        weight_decay = kwarg_parse(kwargs, "weight_decay",1e-1)
        
        return optim.SGD(self.model.parameters(), lr = lr, weight_decay = weight_decay)
    
    def train_step(self, x_train, **kwargs):
        
        b = kwarg_parse(kwargs, "batch_size", 10)
        h = kwarg_parse(kwargs, "horizon", 1)
        
        x,y = self._get_batch(x_train,batch_size = b, horizon = h)
            
        y_hat = self.model(x)
        
        # print(f"y_hat shape: {y_hat.shape}")
            
        l = nn.functional.mse_loss(y,y_hat)
        
        return l
    
    def predict(self, x, horizon = 1):
        
        x = 
        
        t_span = torch.linspace(0,horizon,horizon+1)
        y_hat = self.model(x,t_span)[1][-1,:,-1]
        
        return y_hat
    
    def _get_batch(self,x_train,**kwargs):
        
        batch_size = kwarg_parse(kwargs, "batch_size", 10)
        
        m,d = x_train.shape
        b = min(batch_size, m-1)
        
        #tradeoff between horzion and batch size:
        #choose maximum horizon that provides batch size >= 10
        
        h = kwarg_parse(kwargs,"horizon",(m-b))
        
        h = min(h, m - b)
        
        x_batch = torch.zeros(b,h,d)
        y_batch = torch.zeros(b,h,d)
        
        s = torch.from_numpy(
            np.random.choice(
                np.arange(
                    m-h-1, 
                    dtype = np.int64
                    ),
                    b,
                    replace = False
                    ))
                    
        for i in range(b):
            
            x_batch[i,:,:] = x_train[s[i]:s[i]+h,:]
            y_batch[i,:,:] = x_train[s[i]+1:s[i]+h+1,:]
        
        return x_batch, y_batch
