import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import lightning as L
import re

import ExpMethods.models as m
import ExpMethods.data as data
import ExpMethods.utils as utils
import ExpMethods.simulate as sim
import ExpMethods.visualizations as viz

from ExpMethods.globals import GlobalValues
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
    
    model_dict = {"NODE":None,"LSTM":None}
    
    trainer_params = sim.DefaultSimulationParams.trainer_params(max_epochs = max_epochs)
    
    trainer = L.Trainer(**trainer_params)
 
    sim_params = sim.DefaultSimulationParams.sim_params(
        horizon = max_horizon,
        batch_size = max_batch_size,
        epochs = max_epochs,
        output_dir = args.output_dir)
    
   
    for path in os.listdir(args.input_dir):
        
        id_num = path[path.find("-") + 1: path.find("-") + 4]
        
        pt_dict = utils.get_model_weights(
            model_dict,
            model_dir = args.model_dir,
            id_num = id_num)
        
        sim_params["id_num"] = id_num
        
        df, X = get_data(args.input_dir +"/" + path)
        
        if args.debug:
            df = df.iloc[:100,:] #DEBUGGING ONLY
            X = X[:100] #DEBUGGING ONLY
        
        targets = X[:,-1]
        
        t_end = len(X) - max_horizon
        
        sim_params["end"] = t_end
            
        models = get_model_dict(X, pt_dict, args)
    
        forecasts = sim.get_online_forecasts(models, df, trainer, **sim_params)
        
        losses = sim.get_online_losses(forecasts, targets, **sim_params)
        
        exp_forecasts, exp_losses = sim.get_weighted_forecasts(
            forecasts, 
            losses, 
            targets,
            **sim_params)
        
        regrets = sim.get_regrets(exp_losses, losses)
            
        full_forecasts = forecasts | exp_forecasts
        full_losses = losses | exp_losses
        
        save_to_csv(full_forecasts, full_losses, regrets, id_num, args)
        save_to_png(forecasts,exp_forecasts, losses, exp_losses, regrets, targets, id_num, args)
        
        if args.debug:
            break
    
    return None

def get_data(path):
    
    raw_data = pd.read_csv(path)
    
    df = data.transform_minute_data(raw_data, return_type = pd.DataFrame)
    
    X = torch.tensor(df.to_numpy()).to(torch.float32)
    
    return df, X


def save_to_png(forecasts,exp_forecasts, losses, exp_losses, regrets, targets, id_num, args):
    
    
    viz.plot_forecasts(
        forecasts, 
        targets, 
        title = f"Model Forecasts for Patient {id_num}",
        path = args.output_dir + f"/forecasts/{id_num}_modelforecasts.png")

    viz.plot_forecasts(
        exp_forecasts, 
        targets, 
        title = f"Weighted Forecasts for Patient {id_num}",
        path = args.output_dir + f"/forecasts/{id_num}_expforecasts.png")

    viz.plot_losses(
        losses,
        cumulative = True,
        title = f"Cumulative Model Losses for Patient {id_num}",
        path = args.output_dir + f"/losses/{id_num}_losses.png")

    viz.plot_losses(
        exp_losses,
        cumulative = True,
        title = f"Cumulative Weighted Losses for Patient {id_num}",
        path = args.output_dir + f"/losses/{id_num}_explosses.png")
        
    viz.plot_regrets(
        regrets, 
        title = f"Weighted Regrets for Patient {id_num}",
        path = args.output_dir + f"/regrets/{id_num}_regrets.png")
        
    return None


def save_to_csv(full_forecasts, full_losses, regrets,id_num, args):
    
    utils.save_data(full_forecasts, path = args.output_dir + f"/forecasts/{id_num}_fullforecasts.csv")
    utils.save_data(full_forecasts, path = args.output_dir + f"/losses/{id_num}_fulllosses.csv")
    utils.save_data(regrets, path = args.output_dir + f"/regrets/{id_num}_regrets.csv")
    
    return None

   
def get_model_dict(X, pt_dict, args):
    
    n,d = X.shape
    
    node_params = m.DefaultModelParams.node_params(
        input_dim = d, 
        horizon = args.horizon,
        output_dim = d)
    lstm_params = m.DefaultModelParams.lstm_params(
        input_dim = d, 
        horizon = args.horizon)
    
    base_node = m.NODE(**node_params)
    base_lstm = m.LSTM(**lstm_params)
    
    model_dict = dict(
        NODE = base_node,
        LSTM = base_lstm
    )
    
    for model in model_dict.keys():
        weights = pt_dict.get(model)
        
        if weights is not None:
            
            model_dict[model].load_state_dict(
                torch.load(weights, weights_only = True),
                strict = False
            )
            
    model_dict["NODE"] = m.NODEForecaster(model_dict["NODE"])
    model_dict["LSTM"] = m.LSTMForecaster(model_dict["LSTM"])
            
    return model_dict


def get_args():
    
    parser = argparse.ArgumentParser(
        prog = "Test Forecast Horizons",
        description = "Testing Different Forecast Horizons for Minute Data"
    )
    
    for flag, settings in GlobalValues.command_line_args.items():
        
        parser.add_argument(flag, **settings)
    
    # #data access
    # parser.add_argument("--input_dir",type=str,default="")
    # parser.add_argument("--model_dir",type=str,default="./")
    # parser.add_argument("--output_dir",type=str,default="")
    # 
    # #training params
    # parser.add_argument("--epochs",type=int,default=50)
    # parser.add_argument("--batch",type=int,default=32)
    # parser.add_argument("--t_start",type=int,default=20)
    # parser.add_argument("--tolerance",type=int,default=100)
    # 
    # #simulation params (may also be training params)
    # parser.add_argument("--horizon",type=int,default=30)
    # parser.add_argument("--debug",type=bool,default=False)
    
    ## Add Arguments Specific to Script HERE
    
    args = parser.parse_args()
    return args


def raise_errors(args):
    
    if not args.input_dir:
        raise ValueError("Must Supply Valid Path to Input Data")
    if not args.output_dir:
        raise ValueError("Must Supply Valid Path to Output Data")


if __name__ == "__main__":
    
    main()
