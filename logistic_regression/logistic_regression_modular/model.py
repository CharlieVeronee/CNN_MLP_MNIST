import torch
import numpy as np

#initialize parameters to random value before
#Xavier initialization

#784 inputs: each 28×28 pixel image is flattened into a 784-dimensional vector.
#10outputs: one score (logit) for each of the 10 digit classes (0–9).

def init_params():
    W = torch.randn(784, 10) / np.sqrt(784)
    W.requires_grad_() #we set requires_grad to True to track the gradients
    b = torch.zeros(10, requires_grad=True) # initialize bias b as 0s
    return W, b