import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset

def train_and_record(args, X_train, y_train, X_val, y_val):

    model = NeuralNetwork(args)

    losses = []
    val_accs = []

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

        logits_val = model.forward(X_val)
        preds = np.argmax(logits_val, axis=1)

        if y_val.ndim == 2:
            y_true = np.argmax(y_val, axis=1)
        else:
            y_true = y_val

        acc = accuracy_score(y_true, preds)

        val_accs.append(acc)

        print(f"{args.loss} | Epoch {epoch+1} | Loss {losses[-1]:.4f} | Val Acc {acc:.4f}")

    return losses, val_accs


def plot_curves(mse_loss, ce_loss, mse_acc, ce_acc):

    epochs = np.arange(1, len(mse_loss)+1)

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot(epochs, mse_loss, label="MSE")
    plt.plot(epochs, ce_loss, label="Cross-Entropy")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Loss Convergence")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, mse_acc, label="MSE")
    plt.plot(epochs, ce_acc, label="Cross-Entropy")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()

    X_train, y_train, _, _ = load_dataset("mnist")

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )

    base_args = argparse.Namespace(
        input_dim=784,
        output_dim=10,
        num_layers=3,
        hidden_size=[128,86,64],
        activation="relu",
        weight_init="xavier",
        optimizer="rmsprop",
        learning_rate=0.001,
        weight_decay=0.0,
        batch_size=args.batch_size,
        epochs=args.epochs
    )

    # MSE run
    mse_args = argparse.Namespace(**vars(base_args))
    mse_args.loss = "mse"

    mse_loss, mse_acc = train_and_record(
        mse_args, X_train, y_train, X_val, y_val
    )

    # Cross Entropy run
    ce_args = argparse.Namespace(**vars(base_args))
    ce_args.loss = "cross_entropy"

    ce_loss, ce_acc = train_and_record(
        ce_args, X_train, y_train, X_val, y_val
    )

    plot_curves(mse_loss, ce_loss, mse_acc, ce_acc)


if __name__ == "__main__":
    main()