#!/bin/bash

cd ~/Simulation-Scripts/Hedge-Powered-Predictions

source hedge-env/bin/activate

python ./pretrain_lstm.py \
--input_dir ../daily_data/ \
--model_dir ../pretrained/ \
--epochs 2 \
--horizon 5 \
--n_workers 511 #CHANGE TO 4 FOR GCP RUN \
