import matplotlib.pyplot as plt

#visualize paramter weights for logistic regression model
def visualize_weights(W):
    fig, ax = plt.subplots(1, 10, figsize=(20, 2))
    for i in range(10):
        ax[i].imshow(W[:, i].detach().view(28, 28), cmap='gray')
        ax[i].axis('off')
    plt.show()