import numpy as np
import pandas as pd
import torch
from typing import *


def kwarg_parse(kwargs:dict, key:str, default_val:Any):
    
    return kwargs["key"] if "key" in kwargs.keys() else default_val


def to_np(array: [torch.Tensor,np.ndarray]):
    
    return array if isinstance(array, np.ndarray) else array.detach().cpu().numpy()


def save_data(collection: Dict[str, np.ndarray], **kwargs):
    
    path = kwarg_parse(kwargs, "path", None)
    
    pd.DataFrame(collection).to_csv(path, index = False)
    
    return None
   
    
def make_matrix(collection: Dict[str,np.ndarray]):
    
    return np.hstack(tuple(collection.values()))

