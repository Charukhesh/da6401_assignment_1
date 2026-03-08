import numpy as np
import matplotlib.pyplot as plt

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset
import argparse

def collect_activations(model, X):

    activations = []

    out = X

    for layer in model.layers:

        out = layer.forward(out)

        # collect only activation layers
        if layer.__class__.__name__ in ["ReLU", "Tanh", "Sigmoid"]:
            activations.append(out.copy())

    return activations


def plot_activation_distribution(activations, title):

    num_layers = len(activations)

    plt.figure(figsize=(12, 4))

    for i, act in enumerate(activations):

        plt.subplot(1, num_layers, i + 1)

        plt.hist(act.flatten(), bins=50)

        plt.title(f"Layer {i+1}")

    plt.suptitle(title)

    plt.tight_layout()

    plt.show()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--activation", default="relu")

    args = parser.parse_args()

    X_train, y_train, _, _ = load_dataset("mnist")

    X = X_train[:1024]

    model_args = argparse.Namespace(
        input_dim=784,
        output_dim=10,
        num_layers=3,
        hidden_size=[128, 86, 64],
        activation=args.activation,
        weight_init="xavier",
        weight_decay=0.0,
        loss="cross_entropy",
        optimizer="sgd",
        learning_rate=0.1
    )

    model = NeuralNetwork(model_args)

    activations = collect_activations(model, X)

    plot_activation_distribution(
        activations,
        f"{args.activation.upper()} Activation Distribution"
    )


if __name__ == "__main__":
    main()