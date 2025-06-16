<<<<<<< HEAD
import numpy as np
import pandas as pd
import torch
from typing import *
=======
import os
import numpy as np
import pandas as pd
import torch
import json
import re

from ExpMethods.globals import GlobalValues
from glob import glob

>>>>>>> jsb3

def to_np(array: [torch.Tensor,np.ndarray]):
    
    return array if isinstance(array, np.ndarray) else array.detach().cpu().numpy()


<<<<<<< HEAD
def save_data(collection: Dict[str, np.ndarray], **kwargs):
    
    path = kwargs.get("path", None)
    
    pd.DataFrame(collection).to_csv(path, index = False)
=======
def save_data(collection, **kwargs):
    
    path = kwargs.get("path", None)
    mode = kwargs.get("mode", "w")
    header = kwargs.get("header", True)
    
    pd.DataFrame(collection).to_csv(path, index = False, mode = mode, header = header)
>>>>>>> jsb3
    
    return None


<<<<<<< HEAD
def make_matrix(collection: Dict[str,np.ndarray]):
    
    return np.stack(tuple(collection.values()), axis = 1)
=======
def make_matrix(collection):
    
    return np.stack(tuple(collection.values()), axis = 1)


def get_model_weights(model_dict,**kwargs):
    
    model_dir = kwargs.get("model_dir", "./")
    id_num = kwargs.get("id_num","000")
    
    pt_dict = dict()
    
    model_sort = lambda x: int(re.findall("[0-9]+_[a-z]+_iteration([0-9]+)",x)[0])
    
    for model in filter(lambda m: m.casefold() in GlobalValues.torch_models, model_dict.keys()):
        
        model_name = model.casefold()
        
        dir = os.path.join(model_dir,f"{model_name}")
        
        if not (os.path.exists(dir) or len(glob(os.path.join(dir,"*.pt")))):
            pt_dict[model] = None
        elif not glob(os.path.join(dir,f"{id_num}*.pt")):
            pt_dict[model] = os.path.join(dir,f"pretrained_{model_name}.pt")
        else:
            iteration_models = glob(os.path.join(dir,f"{id_num}*.pt"))
            pt_dict[model] = sorted(iteration_models,key = model_sort)[-1]
        
    return pt_dict


def load_results_from_csv(path, **kwargs):
    
    df = pd.read_csv(path, **kwargs)
    return {col: df[col].to_numpy() for col in df.columns}


def load_targets_from_csv(path, **kwargs):
    
    df = pd.read_csv(path, **kwargs)
    return df.iloc[:,-1].to_numpy()


def save_sim_settings(setting_dict, save_path):
    with open(save_path, "w+") as fp:
        json.dump(setting_dict, fp)

def load_sim_settings(path):
    with open(path, "r") as fp:
        data = json.load(fp)
    return data
>>>>>>> jsb3
