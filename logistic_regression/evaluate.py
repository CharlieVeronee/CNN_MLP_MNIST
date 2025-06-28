import torch
from tqdm.auto import tqdm

def evaluate(test_loader, W, b):
    correct = 0
    total = len(test_loader.dataset)

    with torch.no_grad(): #don't compute gradients again
        #iterate through test set minibatchs 
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            #forward pass
            x = images.view(-1, 28*28)
            y = torch.matmul(x, W) + b # linear transformation with W and b
            predictions = torch.argmax(y, dim=1)
            correct += torch.sum((predictions == labels).float()).item()

    accuracy = correct / total
    return accuracy