#!/bin/bash

cd ~/Hedge-Powered-Predictions

source ../hedge-env/bin/activate

#pip install -r ../requirements.txt

python ./experiment1_horizons.py \
--input_dir ../Data/minute-data/ \
--model_dir ../Outputs/span-test/ \
--output_dir ../Outputs/span-test/ \
--epochs 10 \
--horizon 30 \
--n_workers 4 \
--debug False \

deactivate
