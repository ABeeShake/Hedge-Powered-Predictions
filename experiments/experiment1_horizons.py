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
from statsforecast.models import AutoETS, AutoARIMA 
from xgboost import XGBRegressor

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
        num_workers = args.n_workers,
        log_n_steps = args.log_n_steps,
        context_len = args.context_len)

    node_params = m.DefaultModelParams.node_params(
        horizon = max_horizon)
        
    xgboost_params = m.DefaultModelParams.xgboost_params()
    
    nhits_params = m.DefaultModelParams.nf_params(
        input_size = args.context_len,
        h = max_horizon
        )
        
    autoformer_params = m.DefaultModelParams.nf_params(
        input_size = args.context_len,
        h = max_horizon,
        learning_rate = 0.1
        )
    base_model_dict = {
        "NODE": m.NODE(**node_params),
        "XGBoost": XGBRegressor(**xgboost_params),
        "NHITS": NHITS(**nhits_params),
        "Autoformer": Autoformer(**autoformer_params)
        }
   
    for path in glob(os.path.join(args.input_dir,"*.csv")):

        id_num = os.path.basename(path)[os.path.basename(path).find("-") + 1: os.path.basename(path).find("-") + 4]
        
        existing_forecasts = os.path.join(args.output_dir, "forecasts", f"{id_num}_forecasts.csv")
        
        if os.path.exists(existing_forecasts):
            with open(existing_forecasts,"rb") as f:
                t_start = sum(1 for _ in f)
        
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

        if t_start >= t_end:
            continue
        
        sim_params["end"] = t_end
        
        save_path = os.path.join(args.output_dir, f"settings/{id_num}.json")
        utils.save_sim_settings(sim_params | dir_params, save_path)
        
        sim_params["start"] = t_start
            
        models = get_model_dict(base_model_dict, pt_dict, args)
    
        forecasts = sim.get_online_forecasts(models, X, trainer, **sim_params)
        
        utils.save_data(forecasts, path = os.path.join(args.output_dir, "forecasts", f"{id_num}_forecasts.csv"))
        
        if args.debug:
            break
    
    return None

def get_data(path):
    
    raw_data = pd.read_csv(path)
    
    X = data.transform_minute_data(raw_data)
    
    return X.reshape(-1,1)

   
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
    model_dict["HoltWinters"] = m.StatsForecaster(AutoETS, args.horizon)
    model_dict["ARIMA"] = m.StatsForecaster(AutoARIMA, args.horizon)
    model_dict["XGBoost"] = m.XGBoostForecaster(base_model_dict["XGBoost"], args.horizon)
    model_dict["NHITS"] = m.NNForecaster(base_model_dict["NHITS"])
    model_dict["Autoformer"] = m.NNForecaster(base_model_dict["Autoformer"])
    
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
