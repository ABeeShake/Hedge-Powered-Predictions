import numpy as np
import torch
import matplotlib.pyplot as plt

from typing import *
from ExpMethods.utils import *

def plot_forecasts(forecasts:Dict[str, np.ndarray],targets: [torch.Tensor, np.ndarray], **kwargs):
    
    title = kwargs.get("title", "Model Forecasts")
    show = kwargs.get("show", False)
    save = kwargs.get("save",True)
    path = kwargs.get("path",None)
    
    targets = to_np(targets).flatten()
    
    t = np.arange(len(targets))
    
    plt.close("all")
    plt.figure()
    plt.plot(t, targets, label = "truth", c = "black")
    
    for model in forecasts.keys():
        
        f = forecasts[model]
        plt.plot(t, f, label = model)
        
    plt.legend()
    plt.title(title)
    plt.xlabel("Time Elapsed (min)")
    plt.ylabel("Blood Glucose (mg/dL)")
    
    if show:
        plt.show()
    if save:
        if path is None:
            raise ValueError("Must Supply Path to Save Image")
        plt.savefig(path)
    
    return None


def plot_losses(losses:Dict[str, np.ndarray], **kwargs):
    
    cumulative = kwargs.get( "cumulative", True)
    title = kwargs.get( "title", "Model Losses")
    show = kwargs.get("show", False)
    save = kwargs.get("save",True)
    path = kwargs.get("path",None)
    
    t = np.arange(len(list(losses.values())[0]))
    
    plt.close("all")
    plt.figure()
    
    for model in losses.keys():
        
        l = losses[model]
        if cumulative:
            l = l.cumsum()
        plt.plot(t, l, label = model)
        
    plt.legend()
    plt.title(title)
    plt.xlabel("Time Elapsed (min)")
    plt.ylabel("Blood Glucose (mg/dL)")
    if show:
        plt.show()
    if save:
        if path is None:
            raise ValueError("Must Supply Path to Save Image")
        plt.savefig(path)
    
    return None


def plot_regrets(regrets:Dict[str, np.ndarray], **kwargs):
    
    title = kwargs.get( "title", "Regrets")
    show = kwargs.get("show",False)
    save = kwargs.get("save",True)
    path = kwargs.get("path",None)
    over_t = kwargs.get("over_t", False)
    scale_func = kwargs.get("scale_func",lambda x: x)
    start = kwargs.get("start",0)
    end = kwargs.get("end", len(list(regrets.values())[0]))
    
    t = np.arange(len(list(regrets.values())[0])) + 1
    
    plt.close("all")
    plt.figure()
    if not over_t:
        plt.plot(t[start:end], t[start:end], label = "Linear Regret", c = "black", linewidth = 3, linestyle = "--")
    
    for method in regrets.keys():
        
        r = scale_func(regrets[method])
        if over_t:
            plt.plot(t[start:end], (r/t)[start:end], label = method)
        else:
            plt.plot(t[start:end], r[start:end], label = method)
        
    plt.legend()
    plt.title(title)
    plt.xlabel("Time Elapsed (min)")
    plt.ylabel("Regret (mg/dL)")
    if show:
        plt.show()
    if save:
        if path is None:
            raise ValueError("Must Supply Path to Save Image")
        plt.savefig(path)
    
    return None


def plot_aggregate_regrets(agg_regrets,**kwargs):
    
    title = kwargs.get("title", "Average Regret Over Time Across All Patients")
    show = kwargs.get("show",False)
    save = kwargs.get("save",True)
    path = kwargs.get("path",None)
    omit = kwargs.get("omit",[])
    start = kwargs.get("start",0)
    end = kwargs.get("end", agg_regrets["mean"].shape[0])

    T = np.arange(max_len) + 1
    
    plt.close("all")
    
    avg_r_dict, q5_r_dict, q95_r_dict = agg_regrets.values()
    
    for key in omit:
        del avg_r_dict[key]
        del q5_r_dict[key]
        del q95_r_dict[key]
    
    for i,key in enumerate(avg_r_dict.keys()):
        col = list(plt.cm.tab10(i))
        plt.plot(T[start:end], (avg_r_dict[key]/T)[start:end], label = f"{key}: Average", color=col)
        plt.plot(T[start:end], (q5_r_dict[key]/T)[start:end], ls = "--", color=col, alpha = 0.2)
        plt.plot(T[start:end], (q95_r_dict[key]/T)[start:end], ls = "--", color=col,alpha = 0.2)
        plt.fill_between(T[start:end], y1 = (q5_r_dict[key]/T)[start:end], y2 = (q95_r_dict[key]/T)[start:end], color=col, alpha = 0.2)
    
    plt.xlabel("Time Elapsed (min)")
    plt.ylabel("Regret Over Time ([mg/dL]/min)")
    plt.ylim(0,max([v.max() for v in agg_regrets.values()]))
    plt.title(title)
    plt.legend()
    if show:
        plt.show()
    if save:
        if path is None:
            raise ValueError("Must Supply Path to Save Image")
        plt.savefig(path)
        
        
def bumpchart(df, show_rank_axis= True, rank_axis_distance= 1.1, ax= None, scatter= False, holes= False,line_args= {}, scatter_args= {}, hole_args= {}):
    
    if ax is None:
        left_yaxis= plt.gca()
    else:
        left_yaxis = ax

    # Creating the right axis.
    right_yaxis = left_yaxis.twinx()
    
    axes = [left_yaxis, right_yaxis]
    
    # Creating the far right axis if show_rank_axis is True
    if show_rank_axis:
        far_right_yaxis = left_yaxis.twinx()
        axes.append(far_right_yaxis)
    
    for col in df.columns:
        y = df[col]
        x = df.index.values
        # Plotting blank points on the right axis/axes 
        # so that they line up with the left axis.
        for axis in axes[1:]:
            axis.plot(x, y, alpha= 0)

        left_yaxis.plot(x, y, **line_args, solid_capstyle='round')
        
        # Adding scatter plots
        if scatter:
            left_yaxis.scatter(x, y, **scatter_args)
            
            #Adding see-through holes
            if holes:
                bg_color = left_yaxis.get_facecolor()
                left_yaxis.scatter(x, y, color= bg_color, **hole_args)

    # Number of lines
    lines = len(df.columns)

    y_ticks = [*range(1, lines + 1)]
    
    # Configuring the axes so that they line up well.
    for axis in axes:
        axis.invert_yaxis()
        axis.set_yticks(y_ticks)
        axis.set_ylim((lines + 0.5, 0.5))
    
    # Sorting the labels to match the ranks.
    left_labels = df.iloc[0].sort_values().index
    right_labels = df.iloc[-1].sort_values().index
    
    left_yaxis.set_yticklabels(left_labels)
    right_yaxis.set_yticklabels(right_labels)
    
    # Setting the position of the far right axis so that it doesn't overlap with the right axis
    if show_rank_axis:
        far_right_yaxis.spines["right"].set_position(("axes", rank_axis_distance))
    
    return axes
