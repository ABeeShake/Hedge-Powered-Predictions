import os
import numpy as np
import pandas as pd
import torch
import json
import re

from ExpMethods.globals import GlobalValues
from glob import glob


def to_np(array: [torch.Tensor,np.ndarray]):
    
    return array if isinstance(array, np.ndarray) else array.detach().cpu().numpy()


def save_data(collection, **kwargs):
    
    path = kwargs.get("path", None)
    mode = kwargs.get("mode", "w")
    header = kwargs.get("header", True)
    
    pd.DataFrame(collection).to_csv(path, index = False, mode = mode, header = header)
    
    return None


def make_matrix(collection):
    
    return np.stack(tuple(collection.values()), axis = 1)


def get_model_weights(model_dict,**kwargs):
    
    model_dir = kwargs.get("model_dir", "./")
    id_num = kwargs.get("id_num","000")
    
    pt_dict = dict()
    
    model_sort = lambda x: int(re.findall("[0-9]+_[a-z]+_iteration([0-9]+)",x)[0])
    
    for model in filter(lambda m: m.casefold() in GlobalValues.torch_models, model_dict.keys()):
        
        model_name = model.casefold()
        print(model_name)
        
        dir = os.path.join(model_dir,f"{model_name}")
        
        dir_exists = os.path.exists(dir)
        pretrained_model = glob(os.path.join(dir,f"pretrained_{model_name}.pt"))
        current_models = glob(os.path.join(dir,f"{id_num}*.pt"))
        
        print(not dir_exists or (not current_models and not pretrained_model))
        print(not current_models and pretrained_model)
        
        if not dir_exists or (not current_models and not pretrained_model):
            pt_dict[model] = None
        elif not current_models and pretrained_model:
            pt_dict[model] = pretrained_model[0]
        else:
            pt_dict[model] = sorted(current_models,key = model_sort)[-1]
        
    return pt_dict


def load_results_from_csv(path, **kwargs):
    
    df = pd.read_csv(path, **kwargs)
    return {col: df[col].to_numpy() for col in df.columns}


def load_targets_from_csv(path, **kwargs):
    
    df = pd.read_csv(path, **kwargs)
    return df["Libre.GL" if "Libre.GL" in df.columns else "Dexcom.GL"].to_numpy()


def save_sim_settings(setting_dict, save_path):
    with open(save_path, "w+") as fp:
        json.dump(setting_dict, fp)


def load_sim_settings(path):
    with open(path, "r") as fp:
        data = json.load(fp)
    return data


def get_processed_files(input_dir,output_dir):
    
    filepaths = os.path.join(output_dir, "forecasts","*_forecasts.csv")
    
    input_files = [
        os.path.join(input_dir,f"CGMacros-{os.path.basename(file)[0:3]}-clean.csv") 
        for file 
        in glob(filepaths)
        ]
    
    return input_files
