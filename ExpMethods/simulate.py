<<<<<<< HEAD
=======
import os
>>>>>>> jsb3
import numpy as np
import pandas as pd
import torch
import lightning as L

import ExpMethods.models as m
import ExpMethods.data as data
import ExpMethods.utils as utils

from lightning.pytorch.callbacks import EarlyStopping,ModelCheckpoint
<<<<<<< HEAD
from copy import deepcopy


def get_online_forecasts(models: dict, df: pd.DataFrame, trainer: L.Trainer, **kwargs):
=======
from ExpMethods.globals import GlobalValues


def get_online_forecasts(models: dict, X: pd.DataFrame, trainer: L.Trainer, **kwargs):
>>>>>>> jsb3
    
    h = kwargs.get("max_horizon", 1)
    b = kwargs.get("max_batch_size", 10)
    start = kwargs.get("start", 20)
<<<<<<< HEAD
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
=======
    end = kwargs.get("end", len(X)-h)
    max_epochs = kwargs.get("max_epochs", 1)
    log_n_steps = kwargs.get("log_n_steps", None)
    output_dir = kwargs.get("output_dir","./")
    id_num = kwargs.get("id_num","000")
    num_workers = kwargs.get("num_workers",511)
    
    forecasts = {k:np.zeros(len(X)) for k in models.keys()}
    
    logged_before = False
    
    for t in range(start, end):
        x_train = X[:t]
        x_test = X[t:]
        
        data_params = dict(
            x_train = x_train,
            x_test = x_test,
            batch_size = b,
            max_horizon = h,
            h_first = True,
            num_workers = num_workers
>>>>>>> jsb3
        )
        
        data_module = data.MinuteDataLightningDataModule(**data_params)
        
        for model in models.keys():
            
            trainer.fit(models[model], data_module)
            
<<<<<<< HEAD
            X_t = torch.tensor(x_train.iloc[-1].to_numpy("float32"))
            
            forecasts[model][t + h] = utils.to_np(models[model].predict(X_t)).item()
            
    return forecasts


def get_online_losses(forecasts, targets, start = None, horizon = None):
=======
            X_t = x_train[-1]
            
            forecasts[model][t + h] = utils.to_np(models[model].predict(X_t)).item()
            
        if log_n_steps and (t - start) % log_n_steps == 0:
            
            os.makedirs(os.path.join(output_dir,"forecasts"), exist_ok = True)
            
            output_csv = os.path.join(output_dir,f"forecasts/{id_num}_forecasts.csv")
            
            if not os.path.exists(output_csv) or not logged_before:
                utils.save_data(forecasts, path = output_csv, mode = "a", header = True)
            elif os.path.exists(output_csv) and not logged_before:
                utils.save_data(forecasts, path = output_csv, mode = "w", header = True)
            else: #exists and has been logged before
                utils.save_data(forecasts, path = output_csv, mode = "a", header = False)
            
            logged_before = True
            
            for model in filter(lambda m: m.casefold() in GlobalValues.torch_models, models.keys()):
                
                model_name = model.casefold()
                
                os.makedirs(os.path.join(output_dir,f"{model_name}"), exist_ok = True)
                
                output_model = os.path.join(output_dir,f"{model_name}/{id_num}_{model_name}_iteration{t}.pt")
                
                torch.save(models[model].state_dict(), output_model)
            
    return forecasts


def get_online_losses(forecasts, targets, **kwargs):
    
    start = kwargs.get("start", None)
    horizon = kwargs.get("horizon",None)
>>>>>>> jsb3
    
    if not start:
        raise ValueError("Starting Time-Step Not Supplied")
    if not horizon:
        raise ValueError("Forecasting Horizon Not Supplied")
    
    f_mat = utils.make_matrix(forecasts) # T x n_model
    
    targets = utils.to_np(targets).reshape(-1, 1) #T x 1
        
    l_mat = (f_mat - targets)**2
    
<<<<<<< HEAD
    methods = forecasts.keys()
    
    losses = dict(zip(methods, l_mat.values.T))
=======
    l_mat[:start+horizon] = 0
    
    methods = forecasts.keys()
    
    losses = dict(zip(methods, l_mat.T))
>>>>>>> jsb3
        
    return losses
    

<<<<<<< HEAD
=======
def weighted_forecasts_from_file(settings_path):
    
    settings = utils.load_sim_settings(settings_path)
    
    output_dir = settings.get("output_dir")
    input_dir = settings.get("input_dir")
    id_num = settings.get("id_num")
    
    forecast_path = os.path.join(output_dir, "forecasts",f"{id_num}_forecasts.csv")
    targets_path = os.path.join(input_dir, f"CGMacros-{id_num}-clean.csv")
    
    forecasts = utils.load_results_from_csv(forecast_path)
    targets = utils.load_targets_from_csv(targets_path)
    
    losses = get_online_losses(forecasts, targets, **settings)

    utils.save_data(losses, os.path.join(output_dir, "losses",f"{id_num}_losses.csv"))
    
    return losses

>>>>>>> jsb3

def get_weighted_forecasts(forecasts, losses, targets, **kwargs):
    
    start = kwargs.get("start",None)
    end = kwargs.get("end",None)
<<<<<<< HEAD
    h = kwargs.get("horizon", None)
=======
    from_file = kwargs.get("from_file",False)
    
    if from_file:
        if isinstance(forecasts, str) and isinstance(losses, str) and isinstance(targets, str):
            forecasts = utils.load_results_from_csv(forecasts)
            losses = utils.load_results_from_csv(losses)
            targets = utils.load_targets_from_csv(targets)
        else:
            raise ValueError("please provide forecasts,losses, and targets paths for from_file mode")
        
>>>>>>> jsb3
    n_beta = 4
    
    if not start or not end:
        raise ValueError("Must Have Start and End Times")
<<<<<<< HEAD
    if not h:
        raise ValueError("Must Have Horizon")
    
    y = utils.to_np(targets)
    f_mat = np.stack(tuple(forecasts.values()), axis = 1)
    l_mat = np.stack(tuple(losses.values()), axis = 1)
=======
    
    y = utils.to_np(targets)
    f_mat = utils.make_matrix(forecasts)
    l_mat = utils.make_matrix(losses)
    
    def replace_nans(arr):
        
        return np.where(np.isnan(arr),0,arr)
>>>>>>> jsb3
    
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
        
<<<<<<< HEAD
        W_tilde = W * np.exp(-eta * J * (l - h))
        return W_tilde
=======
        W_tilde = W * np.exp(replace_nans(-eta * J * (l - h)))
        return W_tilde / W_tilde.sum(axis=1).reshape(-1,1)
>>>>>>> jsb3

    
    def mix_update(beta, W):
        #beta: n_beta x 1 x T
        #W: n_beta x T x m
        
        n_beta, T, m = W.shape
        
<<<<<<< HEAD
        return (beta @ W).reshape(n_beta, m) # n_beta x m
=======
        Wt = (beta @ W).reshape(n_beta, m) # n_beta x m
        #print(Wt.shape)
        return Wt
>>>>>>> jsb3

    
    def eta_update(W, J, l, h, Delta, eta):
        
        #W: n_beta x m
        #J: n_beta x m
        #l: 1 x m
        #h: n_beta x 1
        #eta: n_beta x 1
<<<<<<< HEAD
=======
        #Delta: n_beta x 1
>>>>>>> jsb3
        n_beta, m = W.shape
        
        l = l.reshape(1,-1)
        h = h.reshape(-1,1)
        eta = eta.reshape(-1,1)
<<<<<<< HEAD
        
        M_tilde = W * np.exp(-eta * (h + J * (l - h))) #n_beta x m
        
        M = -(1/eta) * np.log(np.sum(M_tilde,axis = 1)).reshape(-1,1) # n_beta x 1
        
        delta = h - M
        Delta = Delta + delta
        
        eta = np.max((1, np.log(m))) / Delta
=======
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
        eta = np.where(Delta == 0, eta, np.max((1, np.log(m))) / Delta)
       # print(eta.shape)
>>>>>>> jsb3
        
        return Delta, eta


<<<<<<< HEAD
    def get_beta(t,T, alpha):
        
        beta = np.zeros(4, 1, T)
=======
    def get_beta(t,T,start,alpha):
        
        beta = np.zeros((4, 1, T))
>>>>>>> jsb3
        
        hedge_mix = np.zeros(T)
        hedge_mix[t] = 1
        beta[0,0,:] = hedge_mix
        
        FS_start_mix = np.zeros(T)
<<<<<<< HEAD
        FS_start_mix[0] = alpha
        FS_start_mix[t] = 1 - alpha
        beta[1,0,:] = FS_start_mix
=======
        FS_start_mix[start] = alpha
        FS_start_mix[t] = 1 - alpha
        beta[1,0,:] = FS_start_mix if t != start else hedge_mix
>>>>>>> jsb3
        
        FS_uniform_mix = alpha * np.ones(T)
        FS_uniform_mix[t] = 1 - alpha
        FS_uniform_mix[:t] = FS_uniform_mix[:t]/t
<<<<<<< HEAD
        beta[2,0,:] = FS_start_mix
=======
        beta[2,0,:] = FS_uniform_mix if t != start else hedge_mix
>>>>>>> jsb3
            
        q = np.arange(t)
        decay = 1 / (t - q)**2
            
        FS_decay_mix = alpha * np.ones(T)
        FS_decay_mix[t] = 1 - alpha
        FS_decay_mix[:t] = alpha * decay * (1 / decay.sum())
<<<<<<< HEAD
        beta[3,0,:] = FS_start_mix
=======
        beta[3,0,:] = FS_decay_mix if t != start else hedge_mix
>>>>>>> jsb3
        
        return beta
        
    T,m = l_mat.shape
    W = np.zeros((n_beta, T, m))
    Wt = np.ones((n_beta, m))
    beta = np.zeros((n_beta, 1, T))
    Delta = np.zeros(n_beta)
<<<<<<< HEAD
    eta = np.ones(n_beta) * 10
=======
    eta = np.ones(n_beta) * 1000
>>>>>>> jsb3
    
    exp_forecasts = np.zeros((T,n_beta))
    exp_losses = np.zeros((T,n_beta))
    
    for t in range(start, end):
        
<<<<<<< HEAD
        alpha = 1 / (t - start + 1)
        
        beta = get_beta(t,T,alpha)
=======
        Wt = Wt / Wt.sum(axis=1).reshape(-1,1)
        #print(f"Wt:{Wt}")
        
        alpha = 1 / (t - start + 2)
        
        beta = get_beta(t,T,start,alpha)
        #print(f"beta:{beta}")
>>>>>>> jsb3
        
        f = f_mat[t]
        l = l_mat[t]
        
<<<<<<< HEAD
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
=======
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
>>>>>>> jsb3
    
    return forecast_dict, loss_dict
        

<<<<<<< HEAD
=======
def weighted_forecasts_from_file(settings_path):
    
    settings = utils.load_sim_settings(settings_path)
    
    output_dir = settings.get("output_dir")
    input_dir = settings.get("input_dir")
    id_num = settings.get("id_num")
    
    forecast_path = os.path.join(output_dir, "forecasts",f"{id_num}_forecasts.csv")
    losses_path = os.path.join(output_dir, "losses",f"{id_num}_losses.csv")
    targets_path = os.path.join(input_dir, f"CGMacros-{id_num}-clean.csv")
    
    forecasts = utils.load_results_from_csv(forecast_path)
    losses = utils.load_results_from_csv(losses_path)
    targets = utils.load_targets_from_csv(targets_path)
    
    exp_forecasts, exp_losses = get_weighted_foreacsts(forecasts, losses, targets, **settings)
    
    utils.save_data(exp_forecasts, os.path.join(output_dir, "forecasts",f"{id_num}_expforecasts.csv"))
    utils.save_data(exp_losses, os.path.join(output_dir, "losses",f"{id_num}_explosses.csv"))
    
    return exp_forecasts, exp_losses


>>>>>>> jsb3
def get_regrets(exp_losses, losses):
    
    l_mat = utils.make_matrix(losses) #T x n_methods
    h_mat = utils.make_matrix(exp_losses) #T x n_models
    
<<<<<<< HEAD
=======
    keep_rows = h_mat.astype(bool).sum(axis=1).astype(bool)
    
    l_mat = l_mat[keep_rows,:]
    h_mat = h_mat[keep_rows,:]
    
>>>>>>> jsb3
    L_mat = l_mat.cumsum(axis=0)
    H_mat = h_mat.cumsum(axis=0)
    
    L_best = L_mat.min(axis=1).reshape(-1,1) # T x 1
    
    R_mat = H_mat - L_best
    
    methods = exp_losses.keys()
<<<<<<< HEAD
    regrets = dict(zip(methods, R_mat.values.T))
    regrets["Best Partition"] = L_best
        
    return regrets
=======
    regrets = dict(zip(methods, R_mat.T))
        
    return regrets


def regrets_from_file(settings_path):
    
    settings = utils.load_sim_settings(settings_path)
    
    output_dir = settings.get("output_dir")
    id_num = settings.get("id_num")
    
    forecast_path = os.path.join(output_dir, "forecasts",f"{id_num}_expforecasts.csv")
    losses_path = os.path.join(output_dir, "losses",f"{id_num}_explosses.csv")
    
    exp_forecasts = utils.load_results_from_csv(forecast_path)
    exp_losses = utils.load_results_from_csv(losses_path)
    
    regrets = get_regrets(forecasts, losses)
    
    utils.save_data(regrets, os.path.join(output_dir, "regrets",f"{id_num}_regrets.csv"))
    
    return regrets


class DefaultSimulationParams:
    
    def sim_params(**kwargs): 
        return GlobalValues.sim_params | kwargs

    def trainer_params(**kwargs): 
        return GlobalValues.trainer_params | kwargs
>>>>>>> jsb3
