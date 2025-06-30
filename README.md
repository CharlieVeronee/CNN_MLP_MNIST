# MNIST Digit Classification with PyTorch

- project implements and compares several neural network architectures on the MNIST dataset, a benchmark dataset of handwritten digits (0–9)

1. Logistic Regression:
   A single fully connected layer mapping pixel values directly to digit classes.

- Input: 784 (28×28 flattened pixels)
- Output: 10 (digit classes)
- Activation: None (uses nn.CrossEntropyLoss)
- Accuracy: ~90%

2. Multilayer Perceptron (MLP)
   A simple feedforward neural network with two hidden layer and ReLU activation.

Architecture:

- 784 → 500 → ReLU → 100 → ReLu → 10
- Activation: ReLU
- Loss: CrossEntropy
- Accuracy: ~92%

3. Convolutional Neural Network (CNN-Mini)
   A small convolutional model using two conv-pool blocks.

Architecture:

Input (1×28×28) →

Conv2d(1, 32, 5x5) → ReLU → MaxPool(2x2) →

Conv2d(32, 64, 5x5) → ReLU → MaxPool(2x2) →

Flatten →

Linear(7×7×64, 256) → ReLU →

Linear(256, 10)

- Accuracy: ~99.0%

4. Convolutional Neural Network (CNN-Max)
   A deeper CNN inspired by VGG-style blocks with stacked convolutions before pooling.

Architecture:

Input (1×28×28) →

Conv2d(1, 32, 5x5) → ReLU →

Conv2d(32, 32, 5x5) → ReLU → MaxPool(2x2) →

Conv2d(32, 64, 5x5) → ReLU →

Conv2d(64, 64, 5x5) → ReLU → MaxPool(2x2) →

Flatten →

Linear(7×7×64, 256) → ReLU →

Linear(256, 10)

- Accuracy: ~99.2%
