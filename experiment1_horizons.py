import os
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

from glob import glob
from ExpMethods.globals import GlobalValues
from lightning.pytorch.callbacks import EarlyStopping,ModelCheckpoint 

def main():
    
    args = get_args()
    
    raise_errors(args)
    
    torch.set_float32_matmul_precision('high')
    
    max_horizon = args.horizon
    max_batch_size = args.batch
    max_epochs = args.epochs
    tol = args.tolerance
    t_start = args.t_start
    
    trainer_params = sim.DefaultSimulationParams.trainer_params(max_epochs = max_epochs)
    
    trainer = L.Trainer(**trainer_params)
    
    dir_params = dict(
        input_dir = args.input_dir,
        output_dir = args.output_dir,
        model_dir = args.model_dir)
 
    sim_params = sim.DefaultSimulationParams.sim_params(
        horizon = max_horizon,
        batch_size = max_batch_size,
        epochs = max_epochs,
        output_dir = args.output_dir,
        num_workers = args.n_workers)

    node_params = m.DefaultModelParams.node_params(
        horizon = max_horizon)
    lstm_params = m.DefaultModelParams.lstm_params(
        horizon = max_horizon)
        
    base_model_dict = {
        "NODE":m.NODE(**node_params),
        "LSTM":m.LSTM(**lstm_params)}
   
    for path in glob(os.path.join(args.input_dir,"*.csv")):
        
        id_num = os.path.basename(path)[os.path.basename(path).find("-") + 1: os.path.basename(path).find("-") + 4]
        
        existing_forecasts = os.path.join(args.output_dir, "forecasts", f"{id_num}_forecasts.csv")
        
        if os.path.exists(existing_forecasts):
            with open(existing_forecasts,"rb") as f:
                sim_params["start"] = sum(1 for _ in f)
        
        pt_dict = utils.get_model_weights(
            base_model_dict,
            model_dir = args.model_dir,
            id_num = id_num)
        
        sim_params["id_num"] = id_num
        
        X = get_data(path)
        
        if args.debug:
            X = X[:100] #DEBUGGING ONLY
        
        targets = X[:,-1]
        
        t_end = len(X) - max_horizon
        
        if sim_params["start"] >= t_end:
            continue
        
        sim_params["end"] = t_end
        
        save_path = os.path.join(args.output_dir, f"settings/{id_num}.json")
        utils.save_sim_settings(sim_params | dir_params, save_path)
            
        models = get_model_dict(base_model_dict, pt_dict, args)
    
        forecasts = sim.get_online_forecasts(models, X, trainer, **sim_params)
        
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
    
    X = data.transform_minute_data(raw_data)
    
    return X.reshape(-1,1)


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

   
def get_model_dict(base_model_dict, pt_dict, args):
        
    model_dict = dict()
    
    for model in base_model_dict.keys():
        weights = pt_dict.get(model)
        
        if weights is not None:
            
            base_model_dict[model].load_state_dict(
                torch.load(weights, weights_only = True),
                strict = False
            )
            
    model_dict["NODE"] = m.NODEForecaster(base_model_dict["NODE"])
    model_dict["LSTM"] = m.LSTMForecaster(base_model_dict["LSTM"])
            
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
