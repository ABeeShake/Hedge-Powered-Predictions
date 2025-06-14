import os
import numpy as np
import pandas as pd
import torch
import lightning as L

import ExpMethods.models as m
import ExpMethods.data as data
import ExpMethods.utils as utils

from lightning.pytorch.callbacks import EarlyStopping,ModelCheckpoint
from copy import deepcopy
from ExpMethods.globals import GlobalValues


def get_online_forecasts(models: dict, df: pd.DataFrame, trainer: L.Trainer, **kwargs):
    
    h = kwargs.get("max_horizon", 1)
    b = kwargs.get("max_batch_size", 10)
    start = kwargs.get("start", 20)
    end = kwargs.get("end", len(df)-h)
    max_epochs = kwargs.get("max_epochs", 1)
    log_n_steps = kwargs.get("log_n_steps", None)
    output_dir = kwargs.get("output_dir","./")
    id_num = kwargs.get("id_num","000")
    
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
            
        if log_n_steps and (t - start) % log_n_steps == 0:
            
            os.makedirs(os.path.join(output_dir,"forecasts"), exist_ok = True)
            
            output_csv = os.path.join(output_dir,f"forecasts/{id_num}_forecasts.csv")
            
            header = os.path.exists(output_csv)
            utils.save_data(forecasts, path = output_csv, mode = "a", header = header)
            
            for model in filter(lambda m: m.casefold() in GlobalValues.torch_models, models.keys()):
                
                model_name = model.casefold()
                
                os.makedirs(os.path.join(output_dir,f"{model_name}"), exist_ok = True)
                
                output_model = os.path.join(output_dir,f"{model_name}/{id_num}_{model_name}_iteration{t}.pt")
                
                torch.save(models[model].state_dict(), output_model)
            
    return forecasts


def get_online_losses(forecasts, targets, **kwargs):
    
    start = kwargs.get("start", None)
    horizon = kwargs.get("horizon",None)
    
    if not start:
        raise ValueError("Starting Time-Step Not Supplied")
    if not horizon:
        raise ValueError("Forecasting Horizon Not Supplied")
    
    f_mat = utils.make_matrix(forecasts) # T x n_model
    
    targets = utils.to_np(targets).reshape(-1, 1) #T x 1
        
    l_mat = (f_mat - targets)**2
    
    methods = forecasts.keys()
    
    losses = dict(zip(methods, l_mat.T))
        
    return losses
    

def get_weighted_forecasts(forecasts, losses, targets, **kwargs):
    
    start = kwargs.get("start",None)
    end = kwargs.get("end",None)
    n_beta = 4
    
    if not start or not end:
        raise ValueError("Must Have Start and End Times")
    
    y = utils.to_np(targets)
    f_mat = np.stack(tuple(forecasts.values()), axis = 1)
    l_mat = np.stack(tuple(losses.values()), axis = 1)
    
    def replace_nans(arr):
        
        return np.where(np.isnan(arr),0,arr)
    
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
        
        W_tilde = W * np.exp(replace_nans(-eta * J * (l - h)))
        return W_tilde / W_tilde.sum(axis=1).reshape(-1,1)

    
    def mix_update(beta, W):
        #beta: n_beta x 1 x T
        #W: n_beta x T x m
        
        n_beta, T, m = W.shape
        
        Wt = (beta @ W).reshape(n_beta, m) # n_beta x m
        #print(Wt.shape)
        return Wt

    
    def eta_update(W, J, l, h, Delta, eta):
        
        #W: n_beta x m
        #J: n_beta x m
        #l: 1 x m
        #h: n_beta x 1
        #eta: n_beta x 1
        #Delta: n_beta x 1
        n_beta, m = W.shape
        
        l = l.reshape(1,-1)
        h = h.reshape(-1,1)
        eta = eta.reshape(-1,1)
        Delta = Delta.reshape(-1,1)
        
        # print(f"eta:{eta}")
        # print(f"l:{l}")
        # print(f"h:{h}")
        # print(f"l-h:{l-h}")
        # print(f"W:{W}")
        
        
        M_tilde = W * np.exp(replace_nans(-eta * (h + J * (l - h)))) #n_beta x m
        
        #print(f"M_tilde:{M_tilde}")
        
        M = replace_nans(-(1/eta) * np.log(np.sum(M_tilde,axis = 1)).reshape(-1,1)) # n_beta x 1
        
        #print(Delta.shape)
        delta = h - M
        Delta = Delta + delta
        #print(Delta.shape)
        
       # print(eta.shape)
        eta = np.max((1, np.log(m))) / Delta
       # print(eta.shape)
        
        return Delta, eta


    def get_beta(t,T,start,alpha):
        
        beta = np.zeros((4, 1, T))
        
        hedge_mix = np.zeros(T)
        hedge_mix[t] = 1
        beta[0,0,:] = hedge_mix
        
        FS_start_mix = np.zeros(T)
        FS_start_mix[start] = alpha
        FS_start_mix[t] = 1 - alpha
        beta[1,0,:] = FS_start_mix if t != start else hedge_mix
        
        FS_uniform_mix = alpha * np.ones(T)
        FS_uniform_mix[t] = 1 - alpha
        FS_uniform_mix[:t] = FS_uniform_mix[:t]/t
        beta[2,0,:] = FS_uniform_mix if t != start else hedge_mix
            
        q = np.arange(t)
        decay = 1 / (t - q)**2
            
        FS_decay_mix = alpha * np.ones(T)
        FS_decay_mix[t] = 1 - alpha
        FS_decay_mix[:t] = alpha * decay * (1 / decay.sum())
        beta[3,0,:] = FS_decay_mix if t != start else hedge_mix
        
        return beta
        
    T,m = l_mat.shape
    W = np.zeros((n_beta, T, m))
    Wt = np.ones((n_beta, m))
    beta = np.zeros((n_beta, 1, T))
    Delta = np.zeros(n_beta)
    eta = np.ones(n_beta) * 1000
    
    exp_forecasts = np.zeros((T,n_beta))
    exp_losses = np.zeros((T,n_beta))
    
    for t in range(start, end):
        
        Wt = Wt / Wt.sum(axis=1).reshape(-1,1)
        #print(f"Wt:{Wt}")
        
        alpha = 1 / (t - start + 2)
        
        beta = get_beta(t,T,start,alpha)
        #print(f"beta:{beta}")
        
        f = f_mat[t]
        l = l_mat[t]
        
        J = get_Jt(Wt)
        
        exp_forecasts[t,:] = J @ f
        
        h = get_ht(Wt, J, l)
        
        exp_losses[t,:] = J @ l
        
        W_tilde = loss_update(Wt,J,l,h, eta)
        #print(f"W_tilde:{W_tilde}")
        W[:,t,:] = W_tilde
        
        Wt = mix_update(beta, W)
        #print(f"Wt @ beta:{Wt}")
        
        #print(eta)
        
        Delta,eta = eta_update(Wt, J, l, h, Delta, eta)

    methods = ["Hedge","FS (Start)", "FS (Uniform)", "FS (Decay)"]
    forecast_dict = dict(zip(methods, exp_forecasts.T))
    loss_dict = dict(zip(methods, exp_losses.T))
    
    return forecast_dict, loss_dict
        

def get_regrets(exp_losses, losses):
    
    l_mat = utils.make_matrix(losses) #T x n_methods
    h_mat = utils.make_matrix(exp_losses) #T x n_models
    
    keep_rows = h_mat.astype(bool).sum(axis=1).astype(bool)
    
    l_mat = l_mat[keep_rows,:]
    h_mat = h_mat[keep_rows,:]
    
    L_mat = l_mat.cumsum(axis=0)
    H_mat = h_mat.cumsum(axis=0)
    
    L_best = L_mat.min(axis=1).reshape(-1,1) # T x 1
    
    R_mat = H_mat - L_best
    
    methods = exp_losses.keys()
    regrets = dict(zip(methods, R_mat.T))
        
    return regrets


class DefaultSimulationParams:
    
    def sim_params(**kwargs): 
        return GlobalValues.sim_params | kwargs

    def trainer_params(**kwargs): 
        return GlobalValues.trainer_params | kwargs
