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

from lightning.pytorch.callbacks import BatchSizeFinder,EarlyStopping,ModelCheckpoint, LearningRateFinder
from copy import deepcopy
from ExpMethods.globals import GlobalValues


def main():
    
    args = get_args()
    
    torch.set_float32_matmul_precision('high')
    
    horizon = args.horizon
    batch_size = args.batch
    epochs = args.epochs
    tol = args.tolerance

    trainer_params = sim.DefaultSimulationParams.trainer_params(
        max_epochs = epochs,
        enable_checkpointing = True)
 
    earlystop_callback = EarlyStopping(
        monitor = "val_loss", 
        stopping_threshold = tol
        )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath = os.path.join(args.model_dir,"lightning"),
        filename = "{epoch}-{val_loss:.2f}",
        save_top_k = 1,
        save_weights_only = True
        )
    checkpoint_callback.FILE_EXTENSION = ".pt"
    batchfind_callback = BatchSizeFinder(init_val = 16)
    lrfind_callback = LearningRateFinder()
    
    trainer_params["callbacks"] = [
        earlystop_callback,
        checkpoint_callback,
        batchfind_callback,
        lrfind_callback
    ]
    
    trainer = L.Trainer(**trainer_params)
    
    lstm_params = m.DefaultModelParams.lstm_params(
        input_dim = 6,
        horizon = horizon)
        
    lstm = m.LSTMForecaster(m.LSTM(**lstm_params))
    
    data_module = data.DailyDataLightningDataModule(
        root = args.input_dir,
        batch_size = batch_size,
        horizon = horizon,
        num_workers = args.n_workers)
        
    trainer.fit(lstm, data_module)
        
    return None


def get_args():
    
    parser = argparse.ArgumentParser(
        prog = "Pretrain LSTM",
        description = "Pretrain LSTM using similar dataset"
    )
    
    for flag, settings in GlobalValues.command_line_args.items():
        
        parser.add_argument(flag, **settings)
    
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    main()
