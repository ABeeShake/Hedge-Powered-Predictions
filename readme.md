# Hedge Powered Predictions

## Overview

This directory hosts the code used to perform real-time glucose forecasting for diabetic patients. The main method in our work is the Hedge algorithm, which is a *mixture of experts* model that follows the predictions of the expert with the best historical accuracy. We also test modifications of Hedge known as Variable Share and Fixed Share. Together, these three methods are known as Exponential Methods.

## To Do

1. Modify model classes to support PyTorch Lightning
2. Add model classes for LSTM, Simple Exponential Smoothing, and Transformer
3. Add scripts for different experiments
	* test various forecast horizons
	* test impact of covariates
	* test regret of Exponential Methods
4. Modify implementation of Exponential methods
