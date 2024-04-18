import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchdyn as dyn
import pytorch_lightning as pl
from torchdyn.core import NeuralODE
import pysindy as ps

def neural_ode(data_dim, device):

    f = nn.Sequential(
        nn.Linear(data_dim, 64),
        nn.Tanh(),
        nn.Linear(64, 32),
        nn.Tanh(),
        nn.Linear(32, data_dim)
    )

    node = NeuralODE(f, sensitivity = "adjoint", solver = "dopri5").to(device)

    return node

def train_node(t, node, X, optimizer = torch.optim.Adam, loss = nn.MSELoss(), lr = 0.01, maxiters = 100, plot = False, tol = 1e-3, y = None):

    loss_hist = np.zeros(maxiters)

    if y == None:

        y = X

    else:

        y = y.reshape(-1,1)


    if X.shape[0] == X.numel():

        X = X.reshape(-1,1)

    opt = optimizer(node.parameters(), lr = lr)

    for i in range(maxiters):

        opt.zero_grad()

        t_eval, preds = node(X, t)
        preds = preds[-1]

        L = loss(preds, y)
        loss_hist[i] = L.item()

        L.backward()
        opt.step()

        if L.item() <= tol:

            break

    if plot:

        plt.plot(loss_hist)

    return None

def node_forecast(node, X):

    if X.shape[0] == X.numel():

        X = X.reshape(-1,1)

    _, preds = node(X)
    preds = preds[-1]

    return preds

def linear_ode(data_dim, device):

    f = nn.Linear(data_dim, data_dim)

    node = NeuralODE(f, sensitivity = "adjoint", solver = "dopri5").to(device)

    return node

def sindy_model(data_dim, order = 2, degree = 3, threshold = 0.2):

    names = ["x" + str(i) for i in range(1,data_dim+1)]

    diff_method = ps.FiniteDifference(order = order)
    feature_lib = ps.PolynomialLibrary(degree = degree)
    opt = ps.STLSQ(threshold = threshold)

    sindy = ps.SINDy(
        differentiation_method = diff_method,
        feature_library = feature_lib,
        optimizer = opt,
        feature_names = names
        )
    
    return sindy

class RNN(nn.Module):

    def __init__(self, data_dim):
        super(RNN, self).__init__()

        self.lstm1 = nn.LSTM(data_dim, 128)
        self.lin1 = nn.Linear(128,64)
        self.lstm2 = nn.LSTM(64,32)
        self.lin2 = nn.Linear(32,data_dim)

    def forward(self, x):

        x,_ = self.lstm1(x)
        x = self.lin1(x)
        x,_ = self.lstm2(x)
        x = self.lin2(x)

        return x



def train_lstm(model,x_train, y_train = None, epochs = 1000, lr = 0.001, tol = 1e-3):

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x_batch = x_train if torch.is_tensor(y_train) else x_train[:-1]
    y_batch = y_train.reshape(-1,1) if torch.is_tensor(y_train) else x_train[1:,:]

    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        if loss <= tol:

            break

    return None

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout = 0.1, max_len = 5000):

        super(PositionalEncoding,self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0,max_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-np.log(10000.0)/d_model))

        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0,1)
        self.register_buffer("pe",pe)

    def forward(self,x):

        x = x + self.pe[:x.size(0),:]
        return self.dropout(x)
    
class TransformerModel(nn.Module):

    def __init__(self, input_dim, d_model = 64, nhead = 4, num_layers = 2, dropout = 0.2):

        super(TransformerModel, self).__init__()

        self.encoder = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoding_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoding_layers, num_layers)
        self.decoder = nn.Linear(d_model, input_dim)

    def forward(self, x):

        x = self.encoder(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.decoder(x[:,-1,:])

        return x
        
def train_transformer(model,x_train, y_train = None, epochs = 1000, lr = 0.001, tol = 1e-3):

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x_batch = x_train if torch.is_tensor(y_train) else x_train[:-1,]
    y_batch = y_train.reshape(-1,1) if torch.is_tensor(y_train) else x_train[1:,:]

    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        if loss <= tol:

            break

    return None