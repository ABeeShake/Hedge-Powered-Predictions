import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import lightning as L

import ExpMethods.models as m
import ExpMethods.data as data
import ExpMethods.utils as utils
import ExpMethods.simulate as sim
import ExpMethods.visualizations as viz

from lightning.pytorch.callbacks import EarlyStopping,ModelCheckpoint 
from copy import deepcopy
from glob import glob
    
def main():
    
    args = get_args()
    
    raise_errors(args)
    
    torch.set_float32_matmul_precision('high')
    
    max_horizon = args.horizon
    max_batch_size = args.batch
    max_epochs = args.epochs
    tol = args.tolerance
    t_start = args.t_start
    
    lstm_weights = get_best_lstm(args)
    
    trainer_params = dict(
        max_epochs = max_epochs,
    #    strategy = "ddp" #(use if multiple GPUs are available)
        accelerator = "auto",
        precision = "16", #(use mixed precision)
        devices = 1,
        log_every_n_steps = 1,
    #    auto_lr_find = True, #(chooses learning rate automatically (DEPRECATED))
        deterministic = True, #(reproducibility)
        enable_progress_bar = False,
        enable_model_summary = False,
        enable_checkpointing = False
    )
    
    trainer = L.Trainer(**trainer_params)
 
    sim_params = dict(
        max_horizon = max_horizon,
        max_batch_size = max_batch_size,
        start = t_start,
        max_epochs = max_epochs,
        )
   
    for path in os.listdir(args.input_dir):
        
        id_num = path[path.find("-") + 1: path.find("-") + 4]
        
        print(id_num)
        print(os.path.join(args.input_dir, path))
        
        df, X = get_data(args.input_dir +"/" + path)
        targets = X[:,-1]
        
        t_end = len(X) - max_horizon
        
        sim_params["end"] = t_end
            
        models = get_model_dict(X, lstm_weights, args)
    
        forecasts = sim.get_online_forecasts(models, df, trainer, **sim_params)
        
        losses = sim.get_online_losses(forecasts, targets, start = t_start, horizon = max_horizon)
        
        exp_forecasts, exp_losses = sim.get_weighted_forecasts(
            forecasts, 
            losses, 
            targets, 
            start = t_start, 
            end = t_end,
            horizon = max_horizon)
        
        regrets = sim.get_regrets(exp_losses, losses)
            
        full_forecasts = forecasts | exp_forecasts
        full_losses = losses | exp_losses
        
        save_to_csv(full_forecasts, full_losses, regrets, id_num, args)
        save_to_png(forecasts,exp_forecasts, losses, exp_losses, regrets, targets, id_num, args)
        
        break
    
    return None

def get_data(path):
    
    raw_data = pd.read_csv(path)
    
    df = data.transform_minute_data(raw_data, return_type = pd.DataFrame)
    
    X = torch.tensor(df.to_numpy()).to(torch.float32)
    
    return df, X


def save_to_png(forecasts,exp_forecasts, losses, exp_losses, regrets, targets, id_num, args):
    
    
    viz.plot_forecasts(
        foreacsts, 
        targets, 
        title = f"Model Forecasts for Patient {id_num}",
        path = args.output_dir + f"/forecasts/{id_num}_modelforecasts.png")

    viz.plot_forecasts(
        exp_foreacsts, 
        targets, 
        title = f"Weighted Forecasts for Patient {id_num}",
        path = args.output_dir + f"/forecasts/{id_num}_expforecasts.png")

    viz.plot_losses(
        losses,
        cumulative = True,
        title = f"Cumulative Model Losses for Patient {id_num}",
        path = args.output_dir + f"/losses/{id_num}_losses.png")

    viz.plot_losses(
        losses,
        cumulative = True,
        title = f"Cumulative Weighted Losses for Patient {id_num}",
        path = args.output_dir + f"/losses/{id_num}_explosses.png")
        
    viz.plot_regrets(
        regrets, 
        title = f"Weighted Regrets for Patient {id_num}",
        path = args.output_dir + f"/regrets/{id_num}_regrets.png")
        
    return None

def save_to_csv(full_foreacsts, full_losses, regrets,id_num, args):
    
    utils.save_data(full_forecasts, path = args.output_dir + f"/forecasts/{id_num}_fullforecasts.csv")
    utils.save_data(full_forecasts, path = args.output_dir + f"/losses/{id_num}_fulllosses.csv")
    utils.save_data(regrets, path = args.output_dir + f"/regrets/{id_num}_regrets.csv")
    
    return None

   
def get_model_dict(X, best_model, args):
    
    n,d = X.shape
    
    node_params = dict(
        input_dim = d,
        hidden_dim = 64,
        output_dim = d,
        horizon = args.horizon,
        sensitivity = "adjoint",
        solver = "dopri5"
        )
    
    lstm_params = dict(
        input_dim = d,
        hidden_dim = 50,
        n_layers = 1,
        horizon = args.horizon
        )
    
    node = m.NODEForecaster(m.NODE(**node_params))
    
    base_lstm = m.LSTM(**lstm_params)
    
    if best_model:
        
        base_lstm.load_state_dict(
            torch.load(best_model, weights_only = True),
            strict = False)
    
    lstm = m.LSTMForecaster(base_lstm)
    
    model_dict = dict(
        node = node,
        lstm = lstm
    )
    
    return model_dict
    
   
def get_best_lstm(args):
    
    existing_models = glob(os.path.join(args.model_dir,"*.pt"))
    existing_models.sort(
        key=lambda x: os.path.getctime(
            os.path.join(
                args.model_dir, 
                x
                )
                )
                )
    best_model = os.path.join(args.model_dir,existing_models[-1]) if existing_models else None


def get_args():
    
    parser = argparse.ArgumentParser(
        prog = "Test Forecast Horizons",
        description = "Testing Different Forecast Horizons for Minute Data"
    )
    
    #data access
    parser.add_argument("--input_dir",type=str,default="")
    parser.add_argument("--model_dir",type=str,default="./")
    parser.add_argument("--output_dir",type=str,default="")
    
    #training params
    parser.add_argument("--epochs",type=int,default=50)
    parser.add_argument("--batch",type=int,default=32)
    parser.add_argument("--t_start",type=int,default=20)
    parser.add_argument("--tolerance",type=int,default=100)
    
    #simulation params (may also be training params)
    parser.add_argument("--horizon",type=int,default=30)
    
    args = parser.parse_args()
    return args


def raise_errors(args):
    
    if not args.input_dir:
        raise ValueError("Must Supply Valid Path to Input Data")
    if not args.output_dir:
        raise ValueError("Must Supply Valid Path to Output Data")

if __name__ == "__main__":
    
    main()
