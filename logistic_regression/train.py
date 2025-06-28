import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

#1. draw a minibatch
#2. zero the gradients in the buffers for W and b
#3. perform the forward pass (compute prediction, calculate loss)
#4. perform the backward pass (compute gradients, perform SGD step)

def train(train_loader, W, b, optimizer):
    for images, labels in tqdm(train_loader, desc="Training"):
        #zero out the gradients
        optimizer.zero_grad()
        #forward pass
        x = images.view(-1, 28*28) #input here is images, but they are so small so convert to vectors (flattening) using view()
            #-1 tells PyTorch to infer this dimension based on the original dimensions and the other specified dimensions
        y = torch.matmul(x, W) + b
        loss = F.cross_entropy(y, labels) #cross-entropy loss combines the softmax operator and cross-entropy into a single operation
        #backward pass
        loss.backward()
        optimizer.step()