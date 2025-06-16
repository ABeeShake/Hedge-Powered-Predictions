#!/usr/bin/bash

cd ~/Hedge-Powered-Predictions

source ~/hedge-env/bin/activate

modeldir=~/Models/lstm_models

testloss=$(ls -v -1 $modeldir/*test*.pt | head -n1 | sed "s/.*test// ; s/[.].*//")


while [ $testloss -ge 1000 ]
do

python ./pretrain_lstm.py \
--data_dir ~/Data/daily-data \
--model_dir ~/Models/lstm_models \
--epochs 100 \
--num_workers 4 \

testloss=$(ls -v -1 $modeldir/*test*.pt | head -n1 | sed "s/.*test// ; s/[.].*//")

done

deactivate

#sudo shutdown -h now
