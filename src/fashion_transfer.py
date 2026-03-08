import numpy as np
import argparse
from sklearn.metrics import accuracy_score
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset


configs = [

    {
        "name": "MNIST Best",
        "hidden_size": [128,64],
        "activation": "relu",
        "optimizer": "rmsprop"
    },

    {
        "name": "Wider Network",
        "hidden_size": [256,128,64],
        "activation": "relu",
        "optimizer": "rmsprop"
    },

    {
        "name": "Stable Gradients",
        "hidden_size": [128,128,128],
        "activation": "tanh",
        "optimizer": "momentum"
    }
]


def run_config(cfg):

    args = argparse.Namespace(
        input_dim=784,
        output_dim=10,
        num_layers=len(cfg["hidden_size"]),
        hidden_size=cfg["hidden_size"],
        activation=cfg["activation"],
        optimizer=cfg["optimizer"],
        learning_rate=0.001,
        loss="cross_entropy",
        weight_init="xavier",
        weight_decay=0.0
    )

    model = NeuralNetwork(args)

    X_train, y_train, X_test, y_test = load_dataset("fashion_mnist")

    model.train(
        X_train,
        y_train,
        epochs=10,
        batch_size=64
    )

    logits = model.forward(X_test)

    preds = np.argmax(logits, axis=1)

    if y_test.ndim == 2:
        y_test = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_test, preds)

    return acc


def main():

    results = []

    for cfg in configs:

        acc = run_config(cfg)

        print(cfg["name"], "Accuracy:", acc)

        results.append((cfg["name"], acc))

    print("\nSummary:")
    for name, acc in results:
        print(name, "->", acc)


if __name__ == "__main__":
    main()