import numpy as np
import pandas as pd
import torch
import lightning as L

import ExpMethods.models as m
import ExpMethods.data as data
import ExpMethods.utils as utils

from lightning.pytorch.callbacks import EarlyStopping,ModelCheckpoint
from copy import deepcopy


def get_online_forecasts(models: dict, df: pd.DataFrame, trainer: L.Trainer, **kwargs):
    
    h = kwargs.get("max_horizon", 1)
    b = kwargs.get("max_batch_size", 10)
    start = kwargs.get("start", 20)
    end = kwargs.get("end", len(df)-h)
    max_epochs = kwargs.get("max_epochs", 1)
    
    forecasts = {k:np.zeros(len(df)) for k in models.keys()}
    
    for t in range(start, end):
        x_train = df.iloc[:t,:]
        x_test = df.iloc[t:,:]
        
        data_params = dict(
            x_train = df.iloc[:t,:],
            x_test = df.iloc[t:,:],
            batch_size = b,
            max_horizon = h,
            h_first = True
        )
        
        data_module = data.MinuteDataLightningDataModule(**data_params)
        
        for model in models.keys():
            
            trainer.fit(models[model], data_module)
            
            X_t = torch.tensor(x_train.iloc[-1].to_numpy("float32"))
            
            forecasts[model][t + h] = utils.to_np(models[model].predict(X_t)).item()
            
    return forecasts


def get_online_losses(forecasts, targets, start = None, horizon = None):
    
    if not start:
        raise ValueError("Starting Time-Step Not Supplied")
    if not horizon:
        raise ValueError("Forecasting Horizon Not Supplied")
    
    f_mat = utils.make_matrix(forecasts) # T x n_model
    
    targets = utils.to_np(targets).reshape(-1, 1) #T x 1
        
    l_mat = (f_mat - targets)**2
    
    methods = forecasts.keys()
    
    losses = dict(zip(methods, l_mat.values.T))
        
    return losses
    


def get_weighted_forecasts(forecasts, losses, targets, **kwargs):
    
    start = kwargs.get("start",None)
    end = kwargs.get("end",None)
    h = kwargs.get("horizon", None)
    n_beta = 4
    
    if not start or not end:
        raise ValueError("Must Have Start and End Times")
    if not h:
        raise ValueError("Must Have Horizon")
    
    y = utils.to_np(targets)
    f_mat = np.stack(tuple(forecasts.values()), axis = 1)
    l_mat = np.stack(tuple(losses.values()), axis = 1)
    
    def get_Jt(W):
        
        # W: n_beta x m
        
        n_beta, m = W.shape
        
        J = np.zeros_like(W)
        idx = (W.cumsum(1) > np.random.rand(n_beta)[:,None]).argmax(1)
        J[np.arange(n_beta), idx] = 1
        
        return J

    
    def get_ht(W, J, l):
        
        # W: n_beta x m
        # J: n_beta x m
        # l: m x 1
        
        l = l.reshape(-1,1)
        
        h = (W * J) @ l
        
        return h #n_beta x 1
 
        
    def loss_update(W,J,l,h, eta):
        
        #W: n_beta x m
        #J: n_beta x m
        #l: 1 x m
        #h: n_beta x 1
        #eta: n_beta x 1
        
        #W_tilde: n_beta x m
        l = l.reshape(1,-1)
        h = h.reshape(-1,1)
        eta = eta.reshape(-1,1)
        
        W_tilde = W * np.exp(-eta * J * (l - h))
        return W_tilde

    
    def mix_update(beta, W):
        #beta: n_beta x 1 x T
        #W: n_beta x T x m
        
        n_beta, T, m = W.shape
        
        return (beta @ W).reshape(n_beta, m) # n_beta x m

    
    def eta_update(W, J, l, h, Delta, eta):
        
        #W: n_beta x m
        #J: n_beta x m
        #l: 1 x m
        #h: n_beta x 1
        #eta: n_beta x 1
        n_beta, m = W.shape
        
        l = l.reshape(1,-1)
        h = h.reshape(-1,1)
        eta = eta.reshape(-1,1)
        
        M_tilde = W * np.exp(-eta * (h + J * (l - h))) #n_beta x m
        
        M = -(1/eta) * np.log(np.sum(M_tilde,axis = 1)).reshape(-1,1) # n_beta x 1
        
        delta = h - M
        Delta = Delta + delta
        
        eta = np.max((1, np.log(m))) / Delta
        
        return Delta, eta


    def get_beta(t,T, alpha):
        
        beta = np.zeros(4, 1, T)
        
        hedge_mix = np.zeros(T)
        hedge_mix[t] = 1
        beta[0,0,:] = hedge_mix
        
        FS_start_mix = np.zeros(T)
        FS_start_mix[0] = alpha
        FS_start_mix[t] = 1 - alpha
        beta[1,0,:] = FS_start_mix
        
        FS_uniform_mix = alpha * np.ones(T)
        FS_uniform_mix[t] = 1 - alpha
        FS_uniform_mix[:t] = FS_uniform_mix[:t]/t
        beta[2,0,:] = FS_start_mix
            
        q = np.arange(t)
        decay = 1 / (t - q)**2
            
        FS_decay_mix = alpha * np.ones(T)
        FS_decay_mix[t] = 1 - alpha
        FS_decay_mix[:t] = alpha * decay * (1 / decay.sum())
        beta[3,0,:] = FS_start_mix
        
        return beta
        
    T,m = l_mat.shape
    W = np.zeros((n_beta, T, m))
    Wt = np.ones((n_beta, m))
    beta = np.zeros((n_beta, 1, T))
    Delta = np.zeros(n_beta)
    eta = np.ones(n_beta) * 10
    
    exp_forecasts = np.zeros((T,n_beta))
    exp_losses = np.zeros((T,n_beta))
    
    for t in range(start, end):
        
        alpha = 1 / (t - start + 1)
        
        beta = get_beta(t,T,alpha)
        
        f = f_mat[t]
        l = l_mat[t]
        
        J = get_jt(Wt)
        
        exp_forecasts[t,:] = J @ f
        
        h = get_ht(W, J, l)
        
        exp_losses[t,:] = h
        
        W_tilde = loss_update(W,J,l,h, eta)
        W[:,t,:] = W_tilde
        
        Wt = mix_update(beta, W)
        
        eta = eta_update(W, J, l, h, Delta, eta)
        
    methods = ["Hedge","FS (Start)", "FS (Uniform)", "FS (Decay)"]
    forecast_dict = dict(zip(methods, exp_forecasts.values.T))
    loss_dict = dict(zip(methods, exp_losses.values.T))
    
    return forecast_dict, loss_dict
        

def get_regrets(exp_losses, losses):
    
    l_mat = utils.make_matrix(losses) #T x n_methods
    h_mat = utils.make_matrix(exp_losses) #T x n_models
    
    L_mat = l_mat.cumsum(axis=0)
    H_mat = h_mat.cumsum(axis=0)
    
    L_best = L_mat.min(axis=1).reshape(-1,1) # T x 1
    
    R_mat = H_mat - L_best
    
    methods = exp_losses.keys()
    regrets = dict(zip(methods, R_mat.values.T))
    regrets["Best Partition"] = L_best
        
    return regrets
