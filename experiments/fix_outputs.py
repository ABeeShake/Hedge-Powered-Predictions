import os
import argparse
import re
import numpy as np
import pandas as pd

import ExpMethods.utils as utils
import ExpMethods.simulate as sim
import ExpMethods.visualizations as viz

from ExpMethods.globals import GlobalValues
from glob import glob
from functools import partial
from ExpMethods.simulate import MixingMethods, AlphaMethods

def main():
    
    args = get_args()
    
    forecast_dir = os.path.join(args.output_dir,"forecasts")
    losses_dir = os.path.join(args.output_dir,"losses")
    regrets_dir = os.path.join(args.output_dir,"regrets")
    settings_dir = os.path.join(args.output_dir,"settings")
    data_dir = args.data_dir if args.data_dir else os.path.join(args.output_dir,"data")

    for file in glob(os.path.join(forecast_dir,"*_forecasts.csv")):

        id_num = os.path.basename(file)[0:3]
        
        if id_num in args.omit_ids:
            continue
        
        ff_path = os.path.join(forecast_dir,f"{id_num}_fullforecasts.csv")
        f_path = os.path.join(forecast_dir,f"{id_num}_forecasts.csv")
        l_path = os.path.join(losses_dir,f"{id_num}_losses.csv")
        r_path = os.path.join(regrets_dir,f"{id_num}_regrets.csv")
        t_path = os.path.join(data_dir, f"{id_num}.csv")
        s_path = os.path.join(settings_dir,f"{id_num}.json")
        
        if os.path.exists(r_path) and not args.overwrite:
            continue
        
        if args.debug:
            print(f"Processing ID: {id_num}")
        
        settings = utils.load_sim_settings(s_path)
        
        f_df = pd.read_csv(f_path)
        
        cols_to_drop = list(
            set(f_df.columns) & (
                {"LSTM","Hedge","FS (Start)","FS (Uniform)","FS (Decay)"} | set(args.omit_models)
                )
                )
        
        f_df.drop(cols_to_drop, axis = 1,inplace = True)
        
        if "HoltWinters" in f_df.columns:
            f_df["ETS"] = f_df.loc[:,"HoltWinters"]
            f_df.drop("HoltWinters", axis=1,inplace=True)
        
        f_df.to_csv(f_path, index = False)
        
        forecasts = {col: f_df[col].to_numpy() for col in f_df.columns}
        targets = utils.load_targets_from_csv(t_path)
        
        if len(forecasts["NODE"]) != len(targets):
            print(f"SKIPPING ID {id_num} DUE TO LENGTH MISMATCH")
            continue
        
        losses = sim.get_online_losses(forecasts,targets, **settings)
        
        utils.save_data(
            losses, 
            path = os.path.join(args.output_dir,"losses",f"{id_num}_losses.csv"))

        method_names = [
            "Hedge (c)",
            "Hedge (d)",
            "Hedge (fd)",
            ]
        
        methods = get_methods(
            settings,
            methods = {
                "Hedge (c)":sim.DefaultSimulationParams.exp_params(
                    start = settings["start"],
                    end = settings["end"]),
                "Hedge (d)":sim.DefaultSimulationParams.exp_params(
                    start = settings["start"],
                    end = settings["end"],),
                "Hedge (fd)":sim.DefaultSimulationParams.exp_params(
                    start = settings["start"],
                    end = settings["end"],)},
            etas = dict(zip(method_names, [args.eta] * len(method_names))),
            mix_funcs = {
                "Hedge (c)":MixingMethods.FS_start_mix,
                "Hedge (d)":MixingMethods.FS_start_mix,
                "Hedge (fd)":MixingMethods.FS_start_mix,
                },
            alpha_funcs = {
                "Hedge (c)":AlphaMethods.constant_alpha,
                "Hedge (d)":AlphaMethods.decreasing_alpha,
                "Hedge (fd)":AlphaMethods.fastdecreasing_alpha,
                })
            
        exp_forecasts, exp_losses = sim.get_weighted_forecasts(forecasts, losses, methods, **settings)
        full_forecasts = forecasts | exp_forecasts
        full_losses = losses | exp_losses
        
        utils.save_data(
            exp_forecasts, 
            path = os.path.join(forecast_dir,f"{id_num}_expforecasts_eta{args.eta}.csv"))
        utils.save_data(
            exp_losses, 
            path = os.path.join(losses_dir,f"{id_num}_explosses_eta{args.eta}.csv"))
        utils.save_data(
            full_forecasts, 
            path = os.path.join(forecast_dir,f"{id_num}_fullforecasts_eta{args.eta}.csv"))
        utils.save_data(
            full_losses, 
            path = os.path.join(losses_dir,f"{id_num}_fulllosses_eta{args.eta}.csv"))
        
        regrets = sim.get_regrets(exp_losses=exp_losses, losses=losses,**settings)
        full_regrets = sim.get_regrets(exp_losses=full_losses, losses=losses,**settings)
        
        utils.save_data(
            regrets, 
            path = os.path.join(regrets_dir,f"{id_num}_regrets_eta{args.eta}.csv"))
        utils.save_data(
            full_regrets, 
            path = os.path.join(regrets_dir,f"{id_num}_fullregrets_eta{args.eta}.csv"))
    
    get_rmse(
        args.output_dir,
        horizon = re.findall("\w+-(\S+)hr/c*", args.output_dir)[0],
        context = re.findall("context-(\S+)hr", args.output_dir)[0],
        eta = args.eta,
        data_dir = args.data_dir)
    
    return None
        

def get_args():
    
    parser = argparse.ArgumentParser(
        prog = "Test Forecast Horizons",
        description = "Testing Different Forecast Horizons for Minute Data"
    )
    
    for flag, settings in GlobalValues.command_line_args.items():
        
        parser.add_argument(flag, **settings)
    
    ## Add Arguments Specific to Script HERE
    
    parser.add_argument("--omit_ids", nargs = "*", type = str, default = [])
    parser.add_argument("--eta", type = int, default = 10)
    parser.add_argument("--omit_models", nargs = "*", type = str)
    parser.add_argument("--overwrite", action = argparse.BooleanOptionalAction)
    
    args = parser.parse_args()
    return args


def get_methods(settings, methods = {}, **kwargs):
    
    eta = kwargs.get("eta", None)
    mix_funcs = kwargs.get("mix_funcs",None)
    alpha_funcs = kwargs.get("alpha_funcs",None)
    
    if not methods:
        methods = {
            "Hedge":sim.DefaultSimulationParams.exp_params(
                start = settings["start"],
                end = settings["end"],
                eta = 10,
                mix_func = MixingMethods.FS_start_mix,
                alpha_func = AlphaMethods.decreasing_alpha),
        }
    
    else:
    
        if eta:
            for method in methods.keys():
                methods[method]["eta"] = etas[method]
        
        if mix_funcs:
            for method in methods.keys():
                methods[method]["mix_func"] = mix_funcs[method]
                
        if alpha_funcs:
            for method in methods.keys():
                methods[method]["alpha_func"] = alpha_funcs[method]
                
        for method in methods.keys():
            methods[method]["start"] = settings["start"]
            methods[method]["end"] = settings["end"]
    
    return methods


def get_rmse(output_dir, **kwargs):
    
    horizon = kwargs.get("horizon")
    context = kwargs.get("context")
    eta = kwargs.get("eta")
    data_dir = kwargs.get("data_dir")
    
    rmse = pd.DataFrame()
    id_nums = []
    
    for file in glob(os.path.join(output_dir,f"losses/*_fulllosses_eta{eta}.csv")):
        id_nums.append(os.path.basename(file)[0:3])
        rmse = pd.concat([rmse,np.sqrt(pd.read_csv(file).mean(axis=0))],axis=1)
        
    rmse = rmse.T
    current_rmse = np.round(rmse.mean(axis=0),2)
    current_std = np.round(rmse.std(axis=0),2)
    
    rmse["id"] = pd.Series(id_nums, dtype = pd.StringDtype)
    
    
    print(f"RMSE for h = {horizon}, c = {context}, eta = {eta}")
    print(current_rmse)
    
    rmse = pd.melt(
        rmse, 
        id_vars=["id"], 
        value_vars = [col for col in rmse.columns if col != "id"], 
        var_name = "model",
        value_name = "rmse")
    main_dir=os.path.relpath(os.path.join(data_dir,os.pardir))
    rmse_path = os.path.join(main_dir,"rmse.csv")
    
    rmse["horizon"] = horizon
    rmse["context"] = context
    rmse["eta"] = eta
    rmse["dataset"] = os.path.basename(main_dir)
                
    if not os.path.exists(rmse_path):
        utils.save_data(rmse, path = rmse_path, mode = "w", header = True)
    else:
        utils.save_data(rmse, path = rmse_path, mode = "a", header = False)
    
    pd.read_csv(rmse_path).drop_duplicates(subset = ["horizon","context","eta","model","id"], keep = "last").to_csv(rmse_path, index = False)
    
if __name__ == "__main__":
    
    main()
