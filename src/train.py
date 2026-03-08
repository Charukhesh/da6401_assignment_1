import argparse
import json
import numpy as np
import wandb

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

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
    parser.add_argument("--hidden_size", nargs="+", default=[128,64])

    parser.add_argument("--activation", default="relu")
    parser.add_argument("--weight_init", default="xavier")

    parser.add_argument("--wandb_project", default="ae22b028-da6401-as1")

    args = parser.parse_args()
    import ast
    if isinstance(args.hidden_size[0], str) and "[" in args.hidden_size[0]:
        args.hidden_size = ast.literal_eval(args.hidden_size[0])
    else:
        args.hidden_size = list(map(int, args.hidden_size))

    return args

def log_dataset_table(X, y):
    """
    Logs a W&B Table with 5 images per class (0–9).
    Assumes X is flattened (784,) MNIST images.
    """

    table = wandb.Table(columns=["class", "image"])

    if y.ndim == 2:
        y = np.argmax(y, axis=1)

    for cls in range(10):
        idx = np.where(y == cls)[0][:5]
        for i in idx:
            img = X[i].reshape(28, 28)
            table.add_data(cls, wandb.Image(img))

    wandb.log({"dataset_samples": table})

def sweep(args):

    wandb_run = wandb.init() 
    config = wandb.config

    args.learning_rate = config.learning_rate
    args.batch_size = config.batch_size
    args.num_layers = len(args.hidden_size)
    args.hidden_size = config.hidden_size
    args.optimizer = config.optimizer
    args.activation = config.activation

    X_train, y_train, _, _ = load_dataset(args.dataset)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1
    )

    model = NeuralNetwork(args)

    model.train(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    logits = model.forward(X_val)
    preds = np.argmax(logits, axis=1)

    if y_val.ndim == 2:
        y_true = np.argmax(y_val, axis=1)
    else:
        y_true = y_val

    val_acc = accuracy_score(y_true, preds)
    val_f1 = f1_score(y_true, preds, average="macro")

    wandb.log({
        "val_acc": val_acc,
        "val_f1": val_f1
    })

    # weights = model.get_weights()

    # np.save("src/best_model.npy", weights)
    # with open("src/best_config.json", "w") as f:
    #    json.dump(vars(args), f)

    wandb.finish()

def main(sweep_mode):
    args = parse_arguments()

    import os
    best_val_f1 = -1
    if os.path.exists("src/best_config.json"):
        with open("src/best_config.json", "r") as f:
            cfg = json.load(f)
            best_val_f1 = cfg.get("best_val_f1", -1)

    print("Current best saved F1:", best_val_f1)

    import ast
    if isinstance(args.hidden_size, str):
        args.hidden_size = ast.literal_eval(args.hidden_size)

    if sweep_mode:
        sweep(args)
        import sys
        sys.exit("Sweeping done")

    wandb_run = wandb.init(project=args.wandb_project, config=vars(args))

    X_train, y_train, _, _ = load_dataset(args.dataset)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1
    )

    log_dataset_table(X_train, y_train)

    model = NeuralNetwork(args)

    # Train model
    model.train(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    logits = model.forward(X_val)
    preds = np.argmax(logits, axis=1)

    if y_val.ndim == 2:
        y_true = np.argmax(y_val, axis=1)
    else:
        y_true = y_val

    val_acc = accuracy_score(y_true, preds)
    val_f1 = f1_score(y_true, preds, average="macro")

    print(
        f"Final | Val Acc {val_acc:.4f} | Val F1 {val_f1:.4f}"
    )

    # Save model only if better than previous runs
    if val_f1 > best_val_f1:

        weights = model.get_weights()
        np.save("src/best_model.npy", weights)

        cfg = vars(args).copy()
        cfg["best_val_f1"] = val_f1

        with open("src/best_config.json", "w") as f:
            json.dump(cfg, f)

        print("New best model saved")
        
    wandb.finish()

if __name__ == "__main__":
    sweep_mode = False
    main(sweep_mode)