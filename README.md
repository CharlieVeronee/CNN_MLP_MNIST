# MNIST Digit Classification with PyTorch

- **Logistic Regression** (single-layer linear model)
- **Multilayer Perceptron (MLP)** (with ReLU activation and hidden layers)
- **Convolutional Neural Network (mini)** Image ->
  convolution (32 3x3 filters) -> nonlinearity (ReLU) -> (2x2 max pool) ->
  convolution (64 3x3 filters) -> nonlinearity (ReLU) -> (2x2 max pool) -> flatten -> fully connected (256 hidden units) -> nonlinearity (ReLU) ->
  fully connected (10 hidden units) -> softmax

- **Convolutional Neural Network (max)** Image ->
  convolution (32 3x3 filters) -> nonlinearity (ReLU) ->
  convolution (32 3x3 filters) -> nonlinearity (ReLU) -> (2x2 max pool) ->
  convolution (64 3x3 filters) -> nonlinearity (ReLU) ->
  convolution (64 3x3 filters) -> nonlinearity (ReLU) -> (2x2 max pool) -> flatten -> fully connected (256 hidden units) -> nonlinearity (ReLU) ->
  fully connected (10 hidden units) -> softmax

## Results

Model Test Accuracy (MNIST)
Logistic Regression: ~90%
MLP (784 → 128 → 10): ~92%
CNN_mini: (dimensions in comments): ~99.0%
CNN_max: (dimensions in comments): ~99.2%
