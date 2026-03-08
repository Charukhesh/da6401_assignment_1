import numpy as np
import matplotlib.pyplot as plt
import argparse

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset


def train_optimizer(optimizer_name):

    args = argparse.Namespace(
        input_dim=784,
        output_dim=10,
        num_layers=3,
        hidden_size=[128,128,128],
        activation="relu",
        weight_init="xavier",
        loss="cross_entropy",
        optimizer=optimizer_name,
        learning_rate=0.001,
        weight_decay=0.0,
        batch_size=64,
        epochs=5
    )

    model = NeuralNetwork(args)

    X_train, y_train, _, _ = load_dataset("mnist")

    losses = []

    n = X_train.shape[0]

    for epoch in range(args.epochs):

        perm = np.random.permutation(n)
        X_train = X_train[perm]
        y_train = y_train[perm]

        batch_losses = []

        for i in range(0, n, args.batch_size):

            X_batch = X_train[i:i+args.batch_size]
            y_batch = y_train[i:i+args.batch_size]

            logits = model.forward(X_batch)

            if y_batch.ndim == 1:
                y_batch = np.eye(logits.shape[1])[y_batch]

            loss = model.loss_fn.forward(y_batch, logits)

            model.backward(y_batch, logits)
            model.update_weights()

            batch_losses.append(loss)

        losses.append(np.mean(batch_losses))

        print(f"{optimizer_name} | Epoch {epoch+1} | Loss {losses[-1]:.4f}")

    return losses


def main():

    optimizers = ["sgd", "momentum", "nag", "rmsprop"]

    plt.figure(figsize=(8,5))

    for opt in optimizers:

        losses = train_optimizer(opt)

        plt.plot(range(1,6), losses, label=opt.upper())

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Optimizer Convergence Comparison (First 5 Epochs)")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()