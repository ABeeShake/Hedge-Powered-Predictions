import numpy as np
import pandas as pd
import torch
import lightning as L

import ExpMethods.models as m
import ExpMethods.data as data
import ExpMethods.utils as utils

from lightning.pytorch.callbacks import EarlyStopping,ModelCheckpoint


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
    
    losses = dict()
    
    targets = utils.to_np(targets)
    
    for model in forecasts.keys():
        
        losses[model] = (forecasts[model] - targets)**2
        
    return losses
    


# class ExponentialWeightForecasts():
#     
#     def __init__(self, forecasts, losses):
#         
#         self.forecasts = forecasts
#         self.losses = losses
#         
#     def Hedge(self):
#         
#         
