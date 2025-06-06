#!/bin/bash

START = "$(ls | sed -e s/[^0-9]//g)"

python pretrain_lstm.py ../data ../lstm_models --start $START