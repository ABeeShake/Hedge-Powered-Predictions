import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L

from torchdyn.core import NeuralODE
from torchdyn.nn import DepthCat
from ExpMethods.utils import *

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
        
    def training_step(self, batch, batch_idx):
        
        x,y = batch
        
        output = self.model(x,self.t_span)[1] #(h_test x b x d)
        y_hat = output[:y.size(1),:,-1].T
        
        loss = torch.nn.functional.mse_loss(y,y_hat)
        
        self.log("loss", loss, prog_bar = True)
        return {"loss":loss}
    
    def forward(self, x, *args):
        
        t_eval = args[0] if args else self.t_span
        
        return self.model(x, t_eval)
    
    def predict(self, x):
        
        with torch.no_grad():
            
            return self.model(x, self.t_span)[1][-1,:,-1]
    

class NODE(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, horizon, **kwargs):
        
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        f = nn.Sequential(
            DepthCat(1),
            nn.Linear(input_dim+1, hidden_dim),
            nn.Tanh(),
            DepthCat(1),
            nn.Linear(hidden_dim + 1, output_dim),
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
    
    def __init__(self, input_dim, hidden_dim, n_layers, horizon, **kwargs):
        
        super().__init__()
        self.hidden_dim = hidden_dim + hidden_dim%2
        self.n_layers = n_layers
        self.horizon = horizon
        self.in_layer = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim//2, n_layers, batch_first = True)
        self.out_layer = nn.Sequential(
            nn.Linear(hidden_dim//2, horizon),
            nn.Softplus())
    
    def forward(self, x, **kwargs):
        
        if len(x.shape) == 1:
            x = x[None, None, :] # b,h_train,d
        elif len(x.shape) == 2:
            x = x[:,None,:]
        
        
        lstm_in = self.in_layer(x)
        lstm_out, _ = self.lstm(lstm_in) #b,h_train,hidden
        y_hat = self.out_layer(lstm_out[:,-1,:]) #b,h_test
        return y_hat


def print_configs(*args):
    
    model_dict = dict(
        node= ["NEURAL ODE:",
        "\t INPUTS:",
        "\t\tf: neural network for ODE (nn.Sequential)",
        "\t\thorizon: forecast horizon (int)",
        "\t KWARGS:",
        "\t\tFORECASTER|from_transfer: whether to load weights from old model (bool)",
        "\t\tFORECASTER|transfer_path: path to old model weights (str/os.path)",
        "\t\tNeual ODE kwargs: sensitivity, solver, rtol, atol"
        ],
        lstm= ["LSTM:",
        "\t INPUTS:",
        "\t\tinput_dim: number of features (int)",
        "\t\thidden_dim: size of hidden state (int)",
        "\t\tn_layers: number of lstm layers (int)"
        "\t\thorizon: forecast horizon (int)",
        "\t KWARGS:",
        "\t\tFORECASTER|from_transfer: whether to load weights from old model (bool)",
        "\t\tFORECASTER|transfer_path: path to old model weights (str/os.path)"
        ]
    )
    
    if not args:
        args = list(model_dict.keys())
    
    display = [m for a in args for m in model_dict[a]]
    
    print("\n".join(display))
