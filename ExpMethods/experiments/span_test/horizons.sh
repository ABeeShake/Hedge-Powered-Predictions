#!/bin/bash

cd ~/Simulation-Scripts/Hedge-Powered-Predictions

#pip install -r ../requirements.txt

python ./experiment1_horizons.py \
--input_dir ../minute_data/ \
--model_dir ../lstm_models/ \
--output_dir ../span_test \
--epochs 2 \
--horizon 30 \