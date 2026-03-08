import argparse
import numpy as np
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset
    
def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="mnist")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--loss", default="cross_entropy")
    parser.add_argument("--optimizer", default="rmsprop")
    parser.add_argument("--learning_rate", type=float, default=0.001)

    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_size", nargs="+", type=int, default=[128, 64])

    parser.add_argument("--activation", default="relu")
    parser.add_argument("--weight_init", default="xavier")

    parser.add_argument("--wandb_project", default="ae22b028-da6401-as1")

    return parser.parse_args()

def main():
    args = parse_arguments()
    _, _, X_test, y_test = load_dataset(args.dataset)

    model = NeuralNetwork(args)

    weights = np.load("src/best_model.npy", allow_pickle=True).item()
    model.set_weights(weights)

    logits = model.forward(X_test)
    preds = np.argmax(logits, axis=1)

    # Handle one-hot labels if present
    if y_test.ndim == 2:
        y_test = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, average="macro")
    recall = recall_score(y_test, preds, average="macro")
    f1 = f1_score(y_test, preds, average="macro")

    print("Accuracy:", acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

if __name__ == "__main__":
    main()