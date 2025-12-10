import os
import numpy as np
import pandas as pd
import torch
import lightning as L

import ExpMethods.models as m
import ExpMethods.data as data
import ExpMethods.utils as utils

from lightning.pytorch.callbacks import EarlyStopping,ModelCheckpoint
from ExpMethods.globals import GlobalValues


def sim_step(model, data_module, trainer, x_train, t, t_min, start):
    
    if model.type == "torch":
        if t % 20 == 0 or t == start:
            trainer.fit(model, data_module)
        return utils.to_np(model.predict(x_train[-1])).item()    
    elif model.type in ["sf","xgboost"]:
        if t % 10 == 0 or t == start:
            model.fit(x_train[t_min:])
        return model.predict(x_train[-1,None])
    elif model.type == "nf":
        if t % 50 == 0 or t == start:
            model.fit(x_train)
        return model.predict(x_train[-1,None])
    

def get_online_forecasts(models: dict, X: pd.DataFrame, trainer: L.Trainer, **kwargs):
    
    h = kwargs.get("max_horizon", 1)
    b = kwargs.get("max_batch_size", 10)
    start = kwargs.get("start", 20)
    end = kwargs.get("end", len(X)-h)
    max_epochs = kwargs.get("max_epochs", 1)
    log_n_steps = kwargs.get("log_n_steps", None)
    output_dir = kwargs.get("output_dir","./")
    id_num = kwargs.get("id_num","000")
    num_workers = kwargs.get("num_workers",511)
    context_len = kwargs.get("context_len",None)
    
    forecasts = {k:np.zeros(len(X)) for k in models.keys()}
    
    logged_before = False
    
    for t in range(start, end):
        
        if (context_len is None) or (context_len > t):
            t_min = 0
        else:
            t_min = t - context_len
        
        x_train = X[:t]
        x_test = X[t:]
        
        data_params = dict(
            x_train = x_train[t_min:],
            x_test = x_test,
            batch_size = b,
            max_horizon = h,
            h_first = True,
            num_workers = num_workers
        )
        
        data_module = data.MinuteDataLightningDataModule(**data_params)
        
        for model in models.keys():
            forecasts[model][t + h] = sim_step(models[model], data_module, trainer,x_train, t, t_min, start)

        if log_n_steps and t != start and not ((t-start) % log_n_steps) or t == end:
            
            os.makedirs(os.path.join(output_dir,"forecasts"), exist_ok = True)
            
            output_csv = os.path.join(output_dir,f"forecasts/{id_num}_forecasts.csv")
            
            start_idx = t+h - log_n_steps if os.path.exists(output_csv) else 0
            current_rows = {k:v[start_idx:t+h] for k,v in forecasts.items()}
            
            if not os.path.exists(output_csv):
                utils.save_data(current_rows, path = output_csv, mode = "w", header = True)
            else: #exists and has been logged before
                utils.save_data(current_rows, path = output_csv, mode = "a", header = False)
            
            for model in models.keys():
                
                if models[model].type == "torch":
                
                    model_name = model.casefold()
                
                    os.makedirs(os.path.join(output_dir,f"{model_name}"), exist_ok = True)
                
                    output_model = os.path.join(output_dir,f"{model_name}/{id_num}_{model_name}_iteration{t}.pt")
                
                    torch.save(models[model].state_dict(), output_model)
            
    return forecasts


def get_online_losses(forecasts, targets, **kwargs):
    
    start = kwargs.get("start", None)
    horizon = kwargs.get("horizon",None)
    
    if start is None:
        raise ValueError("Starting Time-Step Not Supplied")
    if not horizon:
        raise ValueError("Forecasting Horizon Not Supplied")
    
    f_mat = utils.make_matrix(forecasts) # T x n_model
    
    targets = utils.to_np(targets).reshape(-1, 1) #T x 1
        
    l_mat = (f_mat - targets)**2
    
    l_mat[:start+horizon] = 0
    
    methods = forecasts.keys()
    
    losses = dict(zip(methods, l_mat.T))
        
    return losses
    

def losses_from_file(settings_path):
    
    settings = utils.load_sim_settings(settings_path)
    
    output_dir = settings.get("output_dir")
    input_dir = settings.get("input_dir")
    id_num = settings.get("id_num")
    end = settings.get("end")
    horizon = settings.get("horizon")
    
    forecast_path = os.path.join(output_dir, "forecasts",f"{id_num}_forecasts.csv")
    targets_path = os.path.join(input_dir, f"CGMacros-{id_num}-clean.csv")
    
    forecasts = utils.load_results_from_csv(forecast_path)
    targets = utils.load_targets_from_csv(targets_path)[:end+horizon]
    
    losses = get_online_losses(forecasts, targets, **settings)

    utils.save_data(losses, path = os.path.join(output_dir, "losses",f"{id_num}_losses.csv"))
    
    return losses


def weighted_forecast(forecasts, losses,**kwargs):
    
    start = kwargs.get("start",None)
    end = kwargs.get("end",None)
    from_file = kwargs.get("from_file",False)
    mix_func = kwargs.get("mix_func", lambda W,alpha,t,start: W[t+1])
    alpha_func = kwargs.get("alpha_func", lambda alpha,*args,**kwargs: alpha)
    
    if from_file:
        if isinstance(forecasts, str) and isinstance(losses, str) and isinstance(targets, str):
            forecasts = utils.load_results_from_csv(forecasts)
            losses = utils.load_results_from_csv(losses)
            targets = utils.load_targets_from_csv(targets)
        else:
            raise ValueError("please provide forecasts,losses, and targets paths for from_file mode")
    
    if start is None or not end:
        raise ValueError("Must Have Start and End Times")
    
    #y = utils.to_np(targets)
    f_mat = utils.make_matrix(forecasts)
    l_mat = utils.make_matrix(losses)
    
    maxL = l_mat.max()
    
    l_mat = l_mat / maxL #just for testing
    
    T,m = l_mat.shape
    W = np.ones((T+1, m)) / m
    Wt = np.ones(m)
    Delta = 0
    alpha = kwargs.get("alpha", .5)
    eta = kwargs.get("eta",1)
    jt_prev = -1
    cj = 1
    #gamma = 1e-3
    
    L_t = [np.array([]) for _ in range(m)]
    mu_t = np.zeros(m)
    sigma_t = np.zeros(m)
    zt = 0
    
    exp_forecasts = np.zeros(T)
    exp_losses = np.zeros(T)
    
    for t in range(start,end):
        
        f = f_mat[t]
        l = l_mat[t]
        
        if (l == 0).all():
            continue
        
        Wt = Wt / Wt.sum()
        
        #SELECT ARM TO PLAY
        jt = np.random.choice(m, p = Wt)
        
        exp_forecasts[t] = f[jt]
    
        l_jt = l[jt]
        
        exp_losses[t] = l_jt
        
        L_t[jt] = np.append(L_t[jt], l_jt)
        mu_t[jt] = L_t[jt].mean()
        if len(L_t[jt]) > 1:
            sigma_t[jt] = L_t[jt].std()
            zt = (l_jt - mu_t[jt]) / sigma_t[jt]
            
        
        #LOSS UPDATE
        
        Wt_tilde = Wt * np.exp(-eta * l)
        W[t+1] = Wt_tilde.copy()
        
        #MIX UPDATE
            
        Wt = mix_func(W, alpha,t,start)
        
        #ETA UPDATE
        
        #hedge loss
        # ht = l_jt
        #mix loss
        # mt = (-1/eta)*np.log(Wt.T @ np.exp(-eta * (l/(Wt+gamma)) * (np.arange(m) == jt)))
        # 
        # Delta += ht - mt
        # 
        # eta = 1/Delta
        
        #ALPHA UPDATE
        ## switch often early on (alpha \approx 1 for small t)
        ## start switching when loss of chosen arm shoots up
        ## 
        
        # idea 1: alpha should take into account how often an arm was chosen
        # count number of times j_t was selected consecutively (c_j)
        # if j_t == j_{t-1}: alpha = 1/c_j
        # else: alpha = 1 - 1/c_j
        # 
        # idea 2: alpha should be updated when unusual losses are encountered
        # keep track of the average observed losses of each expert
        # keep track of the observed variance of the losses of each expert
        # if the z-score of a new loss >= 2: alpha = 1
        # if the z-score of a new loss <= -2: 
        
        # print(f"jt:{jt}")
        # print(f"cj:{cj}")
        # print(f"jt_prev:{jt_prev}")
        
        cj_tilde = cj * (jt_prev == jt) + 1
        # print(f"cj_tilde:{cj_tilde}")
        
        alpha_params = dict(
            alpha = alpha,
            t = t,
            start = start,
            cj = cj,
            jt = jt,
            jt_prev = jt_prev,
            zt = zt)
        
        alpha = alpha_func(**alpha_params)
        #alpha = np.abs( (jt_prev != jt) - ( 1/(cj) ) )
        #print(f"alpha:{alpha}")
        jt_prev = jt
        cj = cj_tilde
        
        
    return exp_forecasts, exp_losses * maxL


class MixingMethods:
    
    def hedge_mix(W, alpha, t, start,*args, **kwargs):
        return W[t+1]
        
    def FS_start_mix(W,alpha,t, start, *args, **kwargs):
        T1,m = W.shape
        
        beta = np.zeros(T1)
        beta[start] = alpha
        beta[t+1] = 1 - alpha
        
        return beta @ W
    
    def FS_uniform_mix(W,alpha,t, start, *args, **kwargs):
        T1,m = W.shape
        
        beta = np.ones(T1) * (alpha / (t+1))
        beta[t+1] = 1 - alpha
        
        return beta @ W

    def FS_decay_mix(W,alpha,t, start, *args, **kwargs):
        
        theta = kwargs.get("theta", 2)
        
        T1,m = W.shape
        
        decay = 1 / (t+1 - np.arange(t+1))**theta
        
        beta = np.zeros(T1)
        beta[:t+1] = alpha * decay * (1 / decay.sum())
        beta[t+1] = 1 - alpha
        
        return (beta @ W).copy()

    def FS_decay2_mix(W,alpha,t, start, *args, **kwargs):
        rho = kwargs.get("rho", .1)
        
        T1,m = W.shape
        
        decay = (1-rho)*rho**(t - np.arange(t+1))
        
        beta = np.zeros(T1)
        beta[:t+1] = alpha * decay
        beta[t+1] = 1 - alpha
        
        return (beta @ W).copy()


class AlphaMethods:
    def constant_alpha(alpha, *args, **kwargs):
        return alpha
    
    # Decreasing Alpha
    
    def decreasing_alpha(alpha, t = 0, start = 0, *args, **kwargs):
        alpha = 1/(t-start+1)
        return alpha
    
    # Run-Length Alpha
    
    def runlength_alpha(alpha, cj = 1, jt = 0, jt_prev = -1, *args, **kwargs):
        alpha = np.abs( (jt_prev != jt) - ( 1/(cj) ) )
        return alpha
    
    # Loss-Based Alpha
    
    def lossbased_alpha(alpha, zt = 0, *args, **kwargs):
        alpha = norm.cdf(np.abs(zt))
        return alpha
    

def get_weighted_forecasts(forecasts, losses, methods, **kwargs):
    
    exp_forecasts, exp_losses = dict(), dict()
    
    for name, settings in methods.items():
        
        #print(f"Forecasting for {name}")
        
        exp_forecasts[name], exp_losses[name] = weighted_forecast(forecasts, losses, **settings)
    
    return exp_forecasts, exp_losses
        

def get_regrets(exp_losses, losses,**kwargs):
    
    start = kwargs.get("start",20)
    end = kwargs.get("end",len(list(losses.values())[0]))
    
    #print(start)
    #print(end)
    
    l_mat = utils.make_matrix(losses) #T x n_models
    h_mat = utils.make_matrix(exp_losses) #T x n_methods
    
    #print(f"l_mat shape: {l_mat.shape}")
    #print(f"h_mat shape: {h_mat.shape}")
    
    l_mat = l_mat[start:end,:]
    h_mat = h_mat[start:end,:]
    
    #print(f"l_mat shape: {l_mat.shape}")
    #print(f"h_mat shape: {h_mat.shape}")
    
    H_mat = h_mat.cumsum(axis=0)
    
    #print(f"L_mat shape: {L_mat.shape}")
    #print(f"H_mat shape: {H_mat.shape}")
    
    l_best = l_mat.min(axis=1).reshape(-1,1) # T x 1
    L_best = l_best.cumsum(axis=0)
    
    #print(f"L_best shape: {L_best.shape}")
    
    R_mat = H_mat - L_best
    
    #print(f"R_mat shape: {R_mat.shape}")
    
    methods = exp_losses.keys()
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
    
    utils.save_data(regrets, path = os.path.join(output_dir, "regrets",f"{id_num}_regrets.csv"))
    
    return regrets


class DefaultSimulationParams:
    
    def sim_params(**kwargs): 
        return GlobalValues.sim_params | kwargs

    def trainer_params(**kwargs): 
        return GlobalValues.trainer_params | kwargs
    
    def exp_params(**kwargs):
        return GlobalValues.exp_params | kwargs
