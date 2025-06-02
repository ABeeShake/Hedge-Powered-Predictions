import numpy as np
import pandas as pd
import torch
from typing import *

def to_np(array: [torch.Tensor,np.ndarray]):
    
    return array if isinstance(array, np.ndarray) else array.detach().cpu().numpy()


def save_data(collection: Dict[str, np.ndarray], **kwargs):
    
    path = kwargs.get("path", None)
    
    pd.DataFrame(collection).to_csv(path, index = False)
    
    return None


def make_matrix(collection: Dict[str,np.ndarray]):
    
    return np.stack(tuple(collection.values()), axis = 1)
