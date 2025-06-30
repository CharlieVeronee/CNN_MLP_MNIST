import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm.auto import tqdm

class MNIST_Logistic_Regression(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, x):
        return self.lin(x)

#load the data
mnist_train = datasets.MNIST(root="./datasets", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root="./datasets", train=False, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=100, shuffle=False)

#training
#instantiate model
model = MNIST_Logistic_Regression()

#loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

#iterate through train set minibatchs 
for images, labels in tqdm(train_loader):
    #zero out the gradients
    optimizer.zero_grad()
    
    #forward pass
    x = images.view(-1, 28*28)
    y = model(x)
    loss = criterion(y, labels)
    #backward pass
    loss.backward()
    optimizer.step()

#testing
correct = 0
total = len(mnist_test)

with torch.no_grad():
    #iterate through test set minibatchs 
    for images, labels in tqdm(test_loader):
        #forward pass
        x = images.view(-1, 28*28)
        y = model(x)
        
        predictions = torch.argmax(y, dim=1)
        correct += torch.sum((predictions == labels).float())
    
print('Test accuracy: {}'.format(correct/total))