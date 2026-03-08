import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset
import json

def load_model():

    with open("src/best_config.json") as f:
        cfg = json.load(f)

    class Args:
        pass

    args = Args()

    for k, v in cfg.items():
        setattr(args, k, v)

    model = NeuralNetwork(args)

    weights = np.load("src/best_model.npy", allow_pickle=True).item()

    model.set_weights(weights)

    return model

def plot_confusion_matrix(y_true, preds):

    cm = confusion_matrix(y_true, preds)

    plt.figure(figsize=(6,5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    plt.show()

def plot_failures(X, y_true, preds):

    wrong = np.where(preds != y_true)[0][:25]

    plt.figure(figsize=(6,6))

    for i, idx in enumerate(wrong):

        plt.subplot(5,5,i+1)

        plt.imshow(X[idx].reshape(28,28), cmap="gray")

        plt.title(f"T:{y_true[idx]} P:{preds[idx]}")

        plt.axis("off")

    plt.suptitle("Misclassified Digits")

    plt.tight_layout()

    plt.show()

def main():

    X_train, y_train, X_test, y_test = load_dataset("mnist")

    model = load_model()

    logits = model.forward(X_test)

    preds = np.argmax(logits, axis=1)

    if y_test.ndim == 2:
        y_test = np.argmax(y_test, axis=1)

    plot_confusion_matrix(y_test, preds)

    plot_failures(X_test, y_test, preds)

if __name__ == "__main__":
    main()