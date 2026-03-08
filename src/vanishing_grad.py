import numpy as np
import matplotlib.pyplot as plt
import argparse

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset

def gradient_norm_experiment(activation, hidden_layers):

    args = argparse.Namespace(
        input_dim=784,
        output_dim=10,
        num_layers=len(hidden_layers),
        hidden_size=hidden_layers,
        activation=activation,
        weight_init="xavier",
        loss="cross_entropy",
        optimizer="rmsprop",
        learning_rate=0.001,
        weight_decay=0.0,
        batch_size=64
    )

    model = NeuralNetwork(args)

    X_train, y_train, _, _ = load_dataset("mnist")

    grad_norms = []

    iterations = 100
    batch_size = args.batch_size

    for i in range(iterations):

        X_batch = X_train[i*batch_size:(i+1)*batch_size]
        y_batch = y_train[i*batch_size:(i+1)*batch_size]

        logits = model.forward(X_batch)

        if y_batch.ndim == 1:
            y_batch = np.eye(logits.shape[1])[y_batch]

        model.backward(y_batch, logits)

        grad_matrix = model.grad_W[0]

        norm = np.linalg.norm(grad_matrix)

        grad_norms.append(norm)

        model.update_weights()

    return np.array(grad_norms)

def main():

    configs = [
        [128],
        [128, 128, 128],
        [128, 128, 128, 128, 128]
    ]

    plt.figure(figsize=(10,6))

    for hidden in configs:

        relu_norms = gradient_norm_experiment("relu", hidden)
        sigmoid_norms = gradient_norm_experiment("sigmoid", hidden)

        depth = len(hidden)

        plt.plot(sigmoid_norms,
                 label=f"Sigmoid depth={depth}")

        plt.plot(relu_norms,
                 linestyle="--",
                 label=f"ReLU depth={depth}")

    plt.xlabel("Iteration")
    plt.ylabel("Gradient Norm (Layer 1)")
    plt.title("Vanishing Gradient Analysis")
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()