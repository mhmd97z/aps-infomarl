import torch
import numpy as np

def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module
