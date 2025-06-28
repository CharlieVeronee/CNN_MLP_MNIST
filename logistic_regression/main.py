import torch
from model import init_params
from data import load_data
from train import train
from evaluate import evaluate
from utils import visualize_weights

def main():
    train_loader, test_loader = load_data()
    W, b = init_params()
    optimizer = torch.optim.SGD([W, b], lr=0.1)

    train(train_loader, W, b, optimizer)
    accuracy = evaluate(test_loader, W, b)

    print(f"Test accuracy: {accuracy:.4f}")

    visualize_weights(W)

if __name__ == "__main__":
    main()