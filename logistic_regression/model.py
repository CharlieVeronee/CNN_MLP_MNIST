import torch
import numpy as np

#initialize parameters to random value before
#Xavier initialization

def init_params():
    W = torch.randn(784, 10) / np.sqrt(784)
    W.requires_grad_() #we set requires_grad to True to track the gradients
    b = torch.zeros(10, requires_grad=True) # initialize bias b as 0s
    return W, b