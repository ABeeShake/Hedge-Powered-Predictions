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

from ExpMethods.globals import GlobalValues
from lightning.pytorch.callbacks import EarlyStopping,ModelCheckpoint 
from copy import deepcopy


def main():
    
    args = get_args()
    
    max_horizon = args.horizon
    max_batch_size = args.batch
    max_epochs = args.epochs
    tol = args.tolerance

    callbacks = [
    ModelCheckpoint(save_top_k=1, mode="min", monitor="loss"),
    EarlyStopping(monitor = "loss", mode = "min", stopping_threshold = tol)
    ]

    trainer_params = dict(
        max_epochs = max_epochs,
    #    callbacks = callbacks, #(define callbacks)
    #    strategy = "ddp" #(use if multiple GPUs are available)
        accelerator = "auto",
        precision = "16", #(use mixed precision)
        devices = 1,
        log_every_n_steps = 1,
    #    auto_lr_find = True, #(chooses learning rate automatically (DEPRECATED))
        deterministic = True, #(reproducibility)
        enable_progress_bar = False,
        enable_model_summary = False
    )
    
    trainer = L.Trainer(**trainer_params)
    existing_models = os.listdir(args.model_dir)
    
    max_mse = 1e7
    best_model = os.path.join(args.model_dir,existing_models[-1]) if existing_models else None
    
    for i in range(len(os.listdir(args.data_dir))):
        
        df = load_df(os.path.join(args.data_dir, f"pid{i}.csv"))
        X = torch.tensor(df.to_numpy()).to(torch.float32)
        
        N, d = X.shape
        
        lstm_params = m.DefaultModelParams.lstm_params(
            horizon = max_horizon)

        x_train = X[:(4*N)//5]
        x_test = X[(4*N)//5:]
        
        data_module = data.MinuteDataLightningDataModule(
            x_train, 
            x_test,
            batch_size = max_batch_size,
            max_horizon = max_horizon)
        
        base_lstm = m.LSTM(**lstm_params)
        
        if existing_models and best_model:
            
            base_lstm.load_state_dict(
                torch.load(best_model, weights_only = True),
                strict = False)
        
        lstm = m.LSTMForecaster(base_lstm)
            
        
        trainer.fit(lstm, data_module)
        
        y_test = utils.to_np(x_test[:,-1])
        y_hat_test = utils.to_np(lstm.predict(X[N//2 - max_horizon:N - max_horizon]))
        test_mse = ((y_test - y_hat_test)**2).mean()
        print(test_mse)
        
        if test_mse < max_mse:
            
            best_model = os.path.join(args.model_dir,f"{i}.pt")
            torch.save(deepcopy(lstm.state_dict()), best_model)
            existing_models = os.listdir(args.model_dir)
            max_mse = test_mse
        
        if max_mse <= tol:
            return None
        
    return None

def load_df(path):
    
    return pd.read_csv(path, index_col = 0)

def get_args():
    
    parser = argparse.ArgumentParser(
        prog = "Test Forecast Horizons",
        description = "Testing Different Forecast Horizons for Minute Data"
    )
    
    for flag, settings in GlobalValues.command_line_args.items():
        
        parser.add_argument(flag, **settings)
    
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    
    main()
