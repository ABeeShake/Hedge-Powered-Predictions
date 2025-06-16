#!/bin/bash

cd ~/Hedge-Powered-Predictions

source ../hedge-env/bin/activate

#pip install -r ../requirements.txt

python ./experiment1_horizons.py \
--input_dir ../Data/input-data/minute-data/ \
--model_dir ../Data/output-data/span_test/ \
--output_dir ../Data/output-data/span_test/ \
--epochs 10 \
--horizon 30 \
--debug False \

deactivate
