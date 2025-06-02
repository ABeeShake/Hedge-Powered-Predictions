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

from lightning.pytorch.callbacks import EarlyStopping,ModelCheckpoint 


def main():
    
    args = get_args()
    
    max_horizon = 30
    max_batch_size = 32
    max_epochs = 60

    callbacks = [
    ModelCheckpoint(save_top_k=1, mode="min", monitor="loss"),
    EarlyStopping(monitor = "loss", mode = "min", stopping_threshold = 1)
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
    #    enable_progress_bar = False,
    #    enable_model_summary = False
    )
    
    trainer = L.Trainer(**trainer_params)
    
    max_mse = 1e7
    best_model = None
    
    for i in range(len(os.listdir(args.data_dir))):
        
        df = load_df(os.path.join(args.data_dir, f"pid{i}.csv"))
        X = torch.tensor(df.to_numpy()).to(torch.float32)
        
        N, d = X.shape
        
        lstm_params = dict(
        input_dim = d,
        hidden_dim = 50,
        n_layers = 1,
        horizon = max_horizon
        )
        
        from_transfer = bool(len(os.listdir(args.model_dir)))

        x_train = X[:N//2]
        x_test = X[N//2:]
        
        train_df = df.iloc[:N//2,:]
        test_df = df.iloc[N//2:,:]
        
        data_module = data.MinuteDataLightningDataModule(
            train_df, 
            test_df,
            batch_size = max_batch_size,
            max_horizon = max_horizon)
        
        lstm = m.LSTMForecaster(
            m.LSTM(**lstm_params),
            from_transfer = from_transfer,
            transfer_path = best_model)
            
        
        trainer.fit(lstm, data_module)
        
        y_test = utils.to_np(x_test[:,-1])
        y_hat_test = utils.to_np(lstm.predict(X[N//2 - max_horizon:N - h + 1]))
        test_mse = ((y_test - y_hat_test)**2).mean()
        
        if test_mse < max_mse:
            
            best_model = os.path.join(args.model_dir,f"{i}.pt")
            torch.save(lstm.state_dict(), best_model)
            max_mse = test_mse
            
        
        


def load_df(path):
    
    return pd.read_csv(path, index_col = 0)

def get_args():
    
    parser = argparse.ArgumentParser(
        prog = "Pretrain LSTM",
        description = "Pretrain LSTM using similar dataset"
    )
    
    parser.add_argument("data_dir")
    parser.add_argument("model_dir")
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    main()
