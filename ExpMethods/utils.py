import os
import numpy as np
import pandas as pd
import torch

from typing import Dict, Any
from ExpMethods.globals import GlobalValues
from glob import glob

def to_np(array: [torch.Tensor,np.ndarray]):
    
    return array if isinstance(array, np.ndarray) else array.detach().cpu().numpy()


def save_data(collection: Dict[str, np.ndarray], **kwargs):
    
    path = kwargs.get("path", None)
    mode = kwargs.get("mode", "w")
    header = kwargs.get("header", True)
    
    pd.DataFrame(collection).to_csv(path, index = False, mode = mode, header = header)
    
    return None


def make_matrix(collection: Dict[str,np.ndarray]):
    
    return np.stack(tuple(collection.values()), axis = 1)


def get_model_weights(model_dict: Dict[str, Any],**kwargs):
    
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
            pt_dict[model] = iteration_models.sort(key = model_sort)[-1]
        
    return pt_dict
