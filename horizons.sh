#!/bin/bash

cd ~/Simulation-Scripts/Hedge-Powered-Predictions

source hedge-env/bin/activate

#pip install -r ../requirements.txt

python ./experiment1_horizons.py \
--input_dir ../minute_data/ \
--model_dir ../span_test/ \
--output_dir ../span_test/ \
--epochs 2 \
--horizon 5 \
--debug True \

deactivate