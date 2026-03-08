"""
Main Neural Network Model class
Handles model construction, forward/backward propagation and training.
Uses layers, activations, optimizers and loss functions from other modules.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from ann.neural_layer import Dense
from ann.activations import ReLU, Sigmoid, Tanh
from ann.objective_functions import MeanSquaredError, CrossEntropyLoss
from ann.optimizers import SGD, Momentum, NAG, RMSProp

class NeuralNetwork:

    def __init__(self, cli_args=None):
        self.layers = []

        input_dim = getattr(cli_args, "input_dim", 784)
        output_dim = getattr(cli_args, "output_dim", 10)

        hidden_sizes = cli_args.hidden_size
        num_layers = cli_args.num_layers

        if len(hidden_sizes) != num_layers:
            raise ValueError("Length of hidden_size must equal num_layers")

        if cli_args.activation == "relu":
            activation_cls = ReLU
        elif cli_args.activation == "sigmoid":
            activation_cls = Sigmoid
        elif cli_args.activation == "tanh":
            activation_cls = Tanh
        else:
            raise ValueError("Unsupported activation")

        prev_dim = input_dim

        for h in hidden_sizes:

            self.layers.append(
                Dense(prev_dim, h, cli_args.weight_init, cli_args.weight_decay)
            )

            self.layers.append(activation_cls())

            prev_dim = h

        self.layers.append(
            Dense(prev_dim, output_dim, cli_args.weight_init, cli_args.weight_decay)
        )

        if cli_args.loss.lower() == "mse":
            self.loss_fn = MeanSquaredError()
        elif cli_args.loss.lower() == "cross_entropy":
            self.loss_fn = CrossEntropyLoss()

        lr = cli_args.learning_rate

        if cli_args.optimizer.lower() == "sgd":
            self.optimizer = SGD(lr)
        elif cli_args.optimizer.lower() == "momentum":
            self.optimizer = Momentum(lr)
        elif cli_args.optimizer.lower() == "nag":
            self.optimizer = NAG(lr)
        elif cli_args.optimizer.lower() == "rmsprop":
            self.optimizer = RMSProp(lr)

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        out = X
        for layer in self.layers:
            out = layer.forward(out)

        return out

    def backward(self, y_true, logits):
        """
        Performs backward pass and returns grad_W, grad_b
        """

        # compute initial gradient from loss
        grad = self.loss_fn.backward(y_true, logits)
        grad_W_list = []
        grad_b_list = []

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

            if hasattr(layer, "grad_W"):
                grad_W_list.append(layer.grad_W)
                grad_b_list.append(layer.grad_b)

        self.grad_W = grad_W_list[::-1]
        self.grad_b = grad_b_list[::-1]
        return self.grad_W, self.grad_b

    def train(self, X_train, y_train, epochs=1, batch_size=32):
        n = X_train.shape[0]

        for epoch in range(epochs):

            perm = np.random.permutation(n)

            X_train = X_train[perm]
            y_train = y_train[perm]

            losses = []

            for i in range(0, n, batch_size):

                X_batch = X_train[i : i + batch_size]
                y_batch = y_train[i : i + batch_size]

                logits = self.forward(X_batch)
                if logits.ndim == 1:
                     logits = logits.reshape(1, -1)

                if y_batch.ndim == 1:
                    y_batch = np.eye(logits.shape[1])[y_batch]

                loss = self.loss_fn.forward(y_batch, logits)

                self.backward(y_batch, logits)

                self.update_weights()

                losses.append(loss)

            print(f"Epoch {epoch+1} | Loss {np.mean(losses):.4f}")

        return np.mean(losses)

    def update_weights(self):
        self.optimizer.update(self.layers)

    def evaluate(self, X, y):

        logits = self.forward(X)

        if logits.ndim == 1:
            logits = logits.reshape(1, -1)

        preds = np.argmax(logits, axis=1)

        # Handle both one-hot and class index labels
        if y.ndim == 2:
            labels = np.argmax(y, axis=1)
        else:
            labels = y

        acc = np.mean(preds == labels)

        return acc

    def get_weights(self):
        d = {}
        dense_idx = 0

        for layer in self.layers:

            if hasattr(layer, "W"):

                d[f"W{dense_idx}"] = layer.W.copy()
                d[f"b{dense_idx}"] = layer.b.copy()

                dense_idx += 1

        return d

    def set_weights(self, weight_dict):
        dense_idx = 0

        for layer in self.layers:

            if hasattr(layer, "W"):

                w_key = f"W{dense_idx}"
                b_key = f"b{dense_idx}"

                if w_key in weight_dict:
                    layer.W = weight_dict[w_key].copy()

                if b_key in weight_dict:
                    layer.b = weight_dict[b_key].copy()

                dense_idx += 1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="mnist")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--loss", default="cross_entropy")
    parser.add_argument("--optimizer", default="rmsprop")
    parser.add_argument("--learning_rate", type=float, default=0.001)

    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hidden_size", nargs="+", type=int, default=[128,86,64])

    parser.add_argument("--activation", default="relu")
    parser.add_argument("--weight_init", default="xavier")

    parser.add_argument("--wandb_project", default="ae22b028-da6401-as1")

    args = parser.parse_args()
