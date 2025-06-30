import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm.auto import tqdm, trange

#NCHW ordering (batch channels in, height, width)

#input is 100 x 1 x 28 x 28 (batch = 100, channels = 1 bc gray, height, width)
#batch size = 100
class MNIST_CNN_MAX(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2, stride = 1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding=2, stride = 1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2, stride = 1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=5, padding=2, stride = 1)
        self.fc1 = nn.Linear(7*7*64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # conv layer 1
        x = self.conv1(x)
        x = F.relu(x)

        # conv layer 2
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)  # -> 14x14
        
        # conv layer 3
        x = self.conv3(x)
        x = F.relu(x)

        # conv layer 4 (running layer 3 again now with pool)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2) # -> 7x7
        
        # fc layer 1
        x = x.view(-1, 7*7*64)
        x = self.fc1(x)
        x = F.relu(x)
        
        # fc layer 2
        x = self.fc2(x)
        return x

#load the data
mnist_train = datasets.MNIST(root="./datasets", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root="./datasets", train=False, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=100, shuffle=False)

#training
#instantiate model
model = MNIST_CNN_MAX()

#loss and optimizer
criterion = nn.CrossEntropyLoss() #computes softmax for you
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #Adams optimization instead of SGD

#iterate through train set minibatchs 
for epoch in trange(3):
    for images, labels in tqdm(train_loader):
        #zero out the gradients
        optimizer.zero_grad()

        #forward pass
        x = images
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