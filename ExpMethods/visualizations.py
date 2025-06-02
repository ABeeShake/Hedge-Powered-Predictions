import numpy as np
import torch
import matplotlib.pyplot as plt

from typing import *
from ExpMethods.utils import *

def plot_forecasts(forecasts:Dict[str, np.ndarray],targets: [torch.Tensor, np.ndarray], **kwargs):
    
    title = kwargs.get("title", "Model Forecasts")
    
    targets = to_np(targets).flatten()
    
    t = np.arange(len(targets))
    
    plt.close("all")
    plt.figure(figsize = (10,10))
    plt.plot(t, targets, label = "truth", c = "black")
    
    for model in forecasts.keys():
        
        f = forecasts[model]
        plt.plot(t, f, label = model)
        
    plt.legend()
    plt.title(title)
    plt.show()
    
    return None


def plot_losses(losses:Dict[str, np.ndarray], **kwargs):
    
    cumulative = kwargs.get( "cumulative", False)
    title = kwargs.get( "title", "Model Losses")
    
    t = np.arange(len(list(losses.values())[0]))
    
    plt.close("all")
    plt.figure(figsize = (10,10))
    
    for model in losses.keys():
        
        l = losses[model]
        if cumulative:
            l = l.cumsum()
        plt.plot(t, l, label = model)
        
    plt.legend()
    plt.title(title)
    plt.show()
    
    return None


def plot_regrets(regrets:Dict[str, np.ndarray], **kwargs):
    
    cumulative = kwargs.get( "cumulative", False)
    title = kwargs.get( "title", "Regrets")
    
    t = np.arange(len(list(regrets.values())[0]))
    
    plt.close("all")
    plt.figure(figsize = (10,10))
    
    for method in regrets.keys():
        
        r = regrets[model]
        plt.plot(t, r, label = model)
        
    plt.legend()
    plt.title(title)
    plt.show()
    
    return None
