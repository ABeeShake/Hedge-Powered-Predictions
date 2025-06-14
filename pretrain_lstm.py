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
from copy import deepcopy


def main():
    
    args = get_args()
    
    torch.set_float32_matmul_precision('high')
    
    max_horizon = args.horizon
    max_batch_size = args.batch
    max_epochs = args.epochs
    tol = args.tolerance

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
    existing_models = os.listdir(args.model_dir)
    existing_models.sort(
        key=lambda x: os.path.getctime(
            os.path.join(
                args.model_dir, 
                x
                )
                )
                )
                
    max_mse = 1e10
    best_model = os.path.join(args.model_dir,existing_models[-1]) if existing_models else None
    
    print(f"\nCURRENT MODEL WEIGHTS: {best_model}\n")
    
    for i in range(args.start,len(os.listdir(args.data_dir))):
        
        df = load_df(os.path.join(args.data_dir, f"pid{i}.csv"))
        X = torch.tensor(df.to_numpy()).to(torch.float32)
        
        N, d = X.shape
        
        lstm_params = dict(
        input_dim = d,
        hidden_dim = 50,
        n_layers = 1,
        horizon = max_horizon
        )

        x_train = X[:N//2]
        x_test = X[N//2:]
        
        train_df = df.iloc[:N//2,:]
        test_df = df.iloc[N//2:,:]
        
        data_module = data.MinuteDataLightningDataModule(
            train_df, 
            test_df,
            batch_size = max_batch_size,
            max_horizon = max_horizon,
            num_workers = num_workers)
        
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
        print(f"\nTESTING MSE on PID {i} DATASET: {test_mse}\n")
        
        if test_mse < max_mse:
            
            best_model = os.path.join(args.model_dir,f"{i}_test{int(test_mse)}.pt")
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
        prog = "Pretrain LSTM",
        description = "Pretrain LSTM using similar dataset"
    )
    
    parser.add_argument("--data_dir",type=str,default="")
    parser.add_argument("--model_dir",type=str,default="./")
    parser.add_argument("--epochs",type=int,default=50)
    parser.add_argument("--batch",type=int,default=32)
    parser.add_argument("--horizon",type=int,default=30)
    parser.add_argument("--start",type=int,default=0)
    parser.add_argument("--tolerance",type=int,default=100)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    main()
