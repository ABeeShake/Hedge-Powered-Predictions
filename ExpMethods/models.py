import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L

from torchdyn.core import NeuralODE
from torchdyn.nn import DepthCat
from ExpMethods.utils import *
from ExpMethods.globals import GlobalValues
from torch.nn.utils.rnn import PackedSequence, pad_sequence, pack_padded_sequence, pad_packed_sequence, unpack_sequence

class NODEForecaster(L.LightningModule):
    
    def __init__(self, model,**kwargs):
        
        from_transfer = kwargs.get("from_transfer", False)
        transfer_path = kwargs.get("transfer_path", None)
        self.lr = kwargs.get("lr", 1e-3)
        self.weight_decay = kwargs.get("weight_decay", 1e-1)
        
        super().__init__()
        self.model = model
        self.t_span = torch.linspace(0,self.model.horizon, self.model.horizon+1)

    
    def configure_optimizers(self):
        return torch.optim.Adam(
        self.model.parameters(),
        lr = self.lr, 
        weight_decay = self.weight_decay
        )


    def forward(self, x, *args):
        t_eval = args[0] if args else self.t_span
        return self.model(x, t_eval)

        
    def training_step(self, batch, batch_idx):
        
        x,y = batch
        
        output = self.model(x,self.t_span)[1] #(h_test x b x d)
        y_hat = output[:y.size(1),:,-1].T
        
        loss = torch.nn.functional.mse_loss(y,y_hat)
        
        self.log("loss", loss, prog_bar = True)
        return {"loss":loss}

    
    def validation_step(self, batch, batch_idx):
        
        x,y = batch
        
        output = self.model(x,self.t_span)[1] #(h_test x b x d)
        y_hat = output[:y.size(1),:,-1].T
        
        loss = torch.nn.functional.mse_loss(y,y_hat)
        
        self.log("val_loss", loss, prog_bar = True)


    def test_step(self, batch, batch_idx):
        
        x,y = batch
        
        output = self.model(x,self.t_span)[1] #(h_test x b x d)
        y_hat = output[:y.size(1),:,-1].T
        
        loss = torch.nn.functional.mse_loss(y,y_hat)
        
        self.log("test_loss", loss, prog_bar = True)

    
    def predict(self, x):
        with torch.no_grad():
            return self.model(x, self.t_span)[1][-1,:,-1]
    

class NODE(nn.Module):
    
    def __init__(self, hidden_dim, horizon, **kwargs):
        
        super().__init__()
        self.hidden_dim = hidden_dim
        f = nn.Sequential(
            DepthCat(1),
            nn.Linear(1+1, hidden_dim),
            nn.Tanh(),
            DepthCat(1),
            nn.Linear(hidden_dim + 1, 1),
            nn.Softplus()
        )
        self.model = NeuralODE(f,**kwargs)
        self.horizon = horizon
        self.t_span = torch.linspace(0,horizon,horizon+1)
        
    def forward(self, x, *args):
        
        t_eval = args[0] if args else self.t_span
        
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        elif len(x.shape) == 3:
            x = x[:,0,:]
        
        return self.model(x, t_eval)


class LSTMForecaster(L.LightningModule):
    
    def __init__(self, lstm, **kwargs):
        
        super().__init__()
        self.model = lstm
        self.horizon = self.model.horizon
        
        from_transfer = kwargs.get("from_transfer", False)
        transfer_path = kwargs.get("transfer_path", None)
        self.lr = kwargs.get("lr", 1e-3)
        self.weight_decay = kwargs.get("weight_decay", 1e-1)
        
    def forward(self, x, **kwargs):
            
        return self.model(x)
    
    def configure_optimizers(self):
    
        return torch.optim.SGD(
        self.model.parameters(),
        lr = self.lr, 
        weight_decay = self.weight_decay
        )
    
    def training_step(self, batch, batch_idx):
        
        x,y = batch
        #x: b, h_train, d
        #y: b, h_train
        
        y_hat = self.model(x) #b,h_test
        
        loss = nn.functional.mse_loss(y,y_hat[:,:y.size(1)])
        
        self.log("loss", loss, prog_bar = True)
        return {"loss": loss}
    
    def predict(self, x):
        
        with torch.no_grad():
            
            return self.model(x)[:,-1]

    
class LSTM(nn.Module):
    
    def __init__(self, hidden_dim, n_layers, horizon, **kwargs):
        
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.horizon = horizon
        self.lstm = nn.LSTM(1, self.hidden_dim, n_layers, batch_first = True)
        self.out_layer = nn.Sequential(
            nn.Linear(hidden_dim, horizon),
            nn.Softplus())
    
    def forward(self, x, **kwargs):
        
        if len(x.shape) == 1:
            x = x[None, :, None] # b,h_train,d
        elif len(x.shape) == 2:
            x = x[:,:, None]
        
        lstm_out, _ = self.lstm(x) #b,h_train,hidden
        y_hat = self.out_layer(lstm_out[:,-1,:]) #b,h_test
        return y_hat


class DefaultModelParams:
    

    def node_params(**kwargs): 
        return GlobalValues.node_params | kwargs   

    def lstm_params(**kwargs): 
        return GlobalValues.lstm_params | kwargs

