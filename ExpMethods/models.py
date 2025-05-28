import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L

from torchdyn.core import NeuralODE
from torchdyn.nn import DataControl, DepthCat
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from ExpMethods.utils import *

class NODEForecaster(L.LightningModule):
    
    def __init__(self, model, horizon,**kwargs):
        
        from_transfer = kwargs.get("from_transfer", False)
        transfer_path = kwargs.get("transfer_path", None)
        
        super().__init__()
        self.model = model
        self.t_span = torch.linspace(0,horizon, horizon+1)
        
        if from_transfer:
            
            self.model = self.model.load_state_dict(
                torch.load(
                    transfer_path, 
                    weights_only=True
                    )
                    )
    
    def configure_optimizers(self, **kwargs):
        
        lr = kwargs.get("lr", 1e-3)
        weight_decay = kwargs.get("weight_decay", 1e-1)
        
        return torch.optim.Adam(
        self.model.parameters(),
        lr = lr, 
        weight_decay = weight_decay
        )
        
    def training_step(self, batch, batch_idx):
        
        x,y = batch
        
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        y_hat = self.model(x,self.t_span)[1][-1,:,-1]
        
        loss = torch.nn.functional.mse_loss(y,y_hat)
        
        self.log("loss", loss, prog_bar = True)
        return {"loss":loss}
    
    def forward(self, x, *args):
        
        t_eval = args[0] if args else self.t_span
        
        return self.model(x, t_eval)
    

class NODE(nn.Module):
    
    def __init__(self, f, horizon, **kwargs):
        
        super().__init__()
        self.model = NeuralODE(f,**kwargs)
        self.t_span = torch.linspace(0,horizon,horizon+1)
        
    def forward(self, x, *args):
        
        t_eval = args[0] if args else self.t_span
        
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        elif len(x.shape) == 3:
            x = x[:,0,:]
        
        return self.model(x, t_eval)


class LSTMForecaster(L.LightningModule):
    
    def __init__(self, lstm, horizon, **kwargs):
        
        super().__init__()
        self.model = lstm
        self.h = horizon
        
        from_transfer = kwargs.get("from_transfer", False)
        transfer_path = kwargs.get("transfer_path", None)
        
        if from_transfer:
            
            self.model = self.model.load_state_dict(
                torch.load(
                    transfer_path, 
                    weights_only=True
                    )
                    )
        
    def forward(self, x, **kwargs):
        
        h0 = kwargs.get("h", torch.zeros(
            self.model.n_layers, 
            x.size(0),
            self.model.hidden_dim))
        c0 = kwargs.get("c", torch.zeros(
            self.model.n_layers, 
            x.size(0),
            self.model.hidden_dim))
            
        return self.model(x, h = h0, c = c0)
    
    def configure_optimizers(self, **kwargs):
        
        lr = kwargs.get("lr", 1e-3)
        weight_decay = kwargs.get("weight_decay", 1e-1)
        
        return torch.optim.SGD(
        self.model.parameters(),
        lr = lr, 
        weight_decay = weight_decay
        )
    
    def training_step(self, batch, batch_idx):
        
        x,y = batch
        
        h = torch.zeros(
            self.model.n_layers, 
            x.size(0),
            self.model.hidden_dim)
            
        c = torch.zeros(
            self.model.n_layers, 
            x.size(0),
            self.model.hidden_dim)
        
        y_hat, (h, c) = self.model(x, h=h, c=c)
        
        loss = nn.functional.mse_loss(y,y_hat)
        
        self.log("loss", loss, prog_bar = True)
        return {"loss": loss}
    
    
class LSTM(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, n_layers, horizon, **kwargs):
        
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.h = horizon
        self.device = device
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first = True)
        self.out = nn.Softplus(hidden_dim, horizon)
    
    def forward(self, x, **kwargs):
        
        if len(x.shape) == 1:
            x = x[None, None, :]
        
        h = kwargs.get("h", torch.zeros(
            self.n_layers, 
            x.size(0),
            self.hidden_dim))
        c = kwargs.get("c", torch.zeros(
            self.n_layers, 
            x.size(0),
            self.hidden_dim))
            
            
        output, (h, c) = self.lstm(x, (h,c))
        y_hat = self.out(output[:,-1,:])
        return y_hat[:,-1], hn, cn


def print_configs(*args):
    
    model_dict = dict(
        "node": ["NEURAL ODE:",
        "\t INPUTS:",
        "\t\tf: neural network for ODE (nn.Sequential)",
        "\t\thorizon: forecast horizon (int)",
        "\t\tdevice: device for computing (torch.device)",
        "\t KWARGS:",
        "\t\tfrom_transfer: whether to load weights from old model (bool)",
        "\t\ttransfer_path: path to old model weights (str/os.path)",
        "\t\tNeual ODE kwargs: sensitivity, solver, rtol, atol"
        ],
        "lstm": ["LSTM:",
        "\t INPUTS:",
        "\t\tinput_dim: number of features (int)",
        "\t\thidden_dim: size of hidden state (int)",
        "\t\tn_layers: number of lstm layers (int)"
        "\t\thorizon: forecast horizon (int)",
        "\t\tdevice: device for computing (torch.device)"
        "\t KWARGS:",
        "\t\tfrom_transfer: whether to load weights from old model (bool)",
        "\t\ttransfer_path: path to old model weights (str/os.path)"
        ]
    )
    
    if not args:
        args = list(model_dict.keys())
    
    display = [m for a in args for m in model_dict[a]]
    
    print("\n".join(display))
