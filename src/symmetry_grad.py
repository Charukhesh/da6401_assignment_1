import numpy as np
import matplotlib.pyplot as plt
import argparse

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset

def zero_initialize(model):

    for layer in model.layers:
        if hasattr(layer, "W"):
            layer.W[:] = 0
            layer.b[:] = 0

def collect_gradients(args, init_type):

    model = NeuralNetwork(args)

    if init_type == "zeros":
        zero_initialize(model)

    X_train, y_train, _, _ = load_dataset("mnist")

    grads = []

    iterations = 50
    batch_size = args.batch_size

    for i in range(iterations):

        X_batch = X_train[i*batch_size:(i+1)*batch_size]
        y_batch = y_train[i*batch_size:(i+1)*batch_size]

        logits = model.forward(X_batch)

        if y_batch.ndim == 1:
            y_batch = np.eye(logits.shape[1])[y_batch]

        model.backward(y_batch, logits)

        # gradients from first hidden layer
        grad_matrix = model.grad_W[0]

        neuron_grads = grad_matrix[:, :5].mean(axis=0)

        grads.append(neuron_grads)

        model.update_weights()

    return np.array(grads)


def plot_gradients(zero_grads, xavier_grads):

    iterations = np.arange(1, 51)

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)

    for i in range(5):
        plt.plot(iterations, zero_grads[:, i])

    plt.title("Gradient Evolution (Zero Initialization)")
    plt.xlabel("Iteration")
    plt.ylabel("Gradient")

    plt.subplot(1,2,2)

    for i in range(5):
        plt.plot(iterations, xavier_grads[:, i])

    plt.title("Gradient Evolution (Xavier Initialization)")
    plt.xlabel("Iteration")
    plt.ylabel("Gradient")

    plt.tight_layout()
    plt.show()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()

    args.input_dim = 784
    args.output_dim = 10
    args.num_layers = 3
    args.hidden_size = [128,86,64]
    args.activation = "relu"
    args.loss = "cross_entropy"
    args.optimizer = "sgd"
    args.learning_rate = 0.01
    args.weight_decay = 0.0
    args.weight_init = "xavier"

    zero_grads = collect_gradients(args, "zeros")
    xavier_grads = collect_gradients(args, "xavier")

    plot_gradients(zero_grads, xavier_grads)

if __name__ == "__main__":
    main()