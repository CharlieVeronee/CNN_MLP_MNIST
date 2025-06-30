import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm.auto import tqdm, trange

#NCHW ordering (batch channels in, height, width)

#5x5 convolution -> 2x2 max pool -> 5x5 convoltion -> 2x2 max pool -> fully connected ℝ256 -> fully connected to  ℝ10 (prediction)
#use ReLU activation functions for nonlinearities


#sizes:

#input is 100 x 1 x 28 x 28 (batch = 100, channels = 1 bc gray, height, width)
#conv2d (channels in = 1 bc gray, output channels / filters = 32, size of filter = 5x5, padding = 2)
#conv2d (channels in = 32 from prev conv, output channels / filters = 64, size of filter = 5x5, padding = 2)

#batch size = 100
class MNIST_CNN_MINI(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2, stride = 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2, stride = 1)
        self.fc1 = nn.Linear(7*7*64, 256) #input from flattened feature map, 7x7x64 dimensions -> 256 dimensions
        self.fc2 = nn.Linear(256, 10)#256 dimensions -> 10 dimensions (output classes)

    def forward(self, x):
        # conv layer 1
        x = self.conv1(x) #input: 100, 1, 28, 28, output: 100, 32, 28, 28
        x = F.relu(x) #doesnt affect dimensions
        x = F.max_pool2d(x, kernel_size=2)#output: 100, 32, 14, 14
        
        # conv layer 2
        x = self.conv2(x)#input: 100, 32, 14, 14, output: 100, 64, 14, 14
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2) #output: 100, 64, 7, 7
        
        # fc layer 1
        x = x.view(-1, 7*7*64) #convert to 2D tensor
        x = self.fc1(x) #[100, 3136] → [100, 256]
        x = F.relu(x)
        
        # fc layer 2
        x = self.fc2(x)  #[100, 256] → [100, 10]
        return x

#load the data
mnist_train = datasets.MNIST(root="./datasets", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root="./datasets", train=False, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=100, shuffle=False)

#training
#instantiate model  
model = MNIST_CNN_MINI()

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #Adams optimization instead of SGD

#iterate through train set minibatchs 
for epoch in trange(3):
    for images, labels in tqdm(train_loader):
        #zero out the gradients
        optimizer.zero_grad()

        #forward pass
        x = images #don't need to reshape images, already 4D
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
        x = images
        y = model(x)

        predictions = torch.argmax(y, dim=1)
        correct += torch.sum((predictions == labels).float())

print('Test accuracy: {}'.format(correct/total))