import torch, json, os
import pandas as pd
import numpy as np
from jiwer import cer, wer
from tqdm import tqdm
from einops import rearrange

def load_dictionary(dict_path):
    with open(dict_path, 'r') as file:
        dictionary = json.load(file)
        
    return dictionary

def flip_dictionary(dictionary):
    
    flipped_dictionary = {v: k for k, v in dictionary.items()}
    
    return flipped_dictionary

def normalize_audio(x):
    max_value = torch.max(np.abs(x))

    normalized_x = x / max_value
    
    return normalized_x