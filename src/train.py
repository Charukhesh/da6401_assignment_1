import argparse
import json
import numpy as np
import wandb

from sklearn.model_selection import train_test_split

from ann.neural_network import NeuralNetwork
from ann.neural_layer import Dense
from ann.activations import ReLU, Sigmoid, Tanh
from ann.objective_functions import CrossEntropyLoss, MeanSquaredError
from ann.optimizers import SGD, Momentum, NAG, RMSProp
from ann.training import train
from utils.data_loader import load_dataset

def get_activation(name):
    if name == "relu":
        return ReLU()
    elif name == "sigmoid":
        return Sigmoid()
    elif name == "tanh":
        return Tanh()
    else:
        raise ValueError("Unsupported activation")
    
def get_optimizer(name, lr):
    if name == "sgd":
        return SGD(lr)
    elif name == "momentum":
        return Momentum(lr)
    elif name == "nag":
        return NAG(lr)
    elif name == "rmsprop":
        return RMSProp(lr)
    else:
        raise ValueError("Unsupported optimiser")
    
def get_loss(name):
    if name == "cross_entropy":
        return CrossEntropyLoss()
    elif name == "mean_squared_error":
        return MeanSquaredError()
    else:
        raise ValueError("Unsupported loss")
    
def build_model(input_dim, hidden_sizes, activation, weight_init, weight_decay):
    model = NeuralNetwork()
    prev_dim = input_dim

    for h in hidden_sizes:
        model.add(Dense(prev_dim, h, weight_init, weight_decay))
        model.add(get_activation(activation))
        prev_dim = h

    model.add(Dense(prev_dim, 10, weight_init, weight_decay))
    return model

def log_dataset_samples(X, y):
    grid = []

    for cls in range(10):
        idx = np.where(y == cls)[0][:5]
        row = [X[i].reshape(28,28) for i in idx]
        grid.append(np.concatenate(row, axis=1))

    grid_image = np.concatenate(grid, axis=0)

    wandb.log({
        "dataset_samples": wandb.Image(grid_image, caption="5 samples per class")
    })

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="mnist")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--loss", default="cross_entropy")
    parser.add_argument("--optimizer", default="sgd")
    parser.add_argument("--learning_rate", type=float, default=0.01)

    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_size", nargs="+", type=int, default=[128, 64])

    parser.add_argument("--activation", default="relu")
    parser.add_argument("--weight_init", default="xavier")

    parser.add_argument("--wandb_project", default="ae22b028-da6401-as1")

    args = parser.parse_args()

    wandb_run = wandb.init(project=args.wandb_project, config=vars(args))

    X_train, y_train, _, _ = load_dataset(args.dataset)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)
    log_dataset_samples(X_train, y_train)

    hidden_sizes = args.hidden_size[: args.num_layers]

    model = build_model(
        input_dim=784,
        hidden_sizes=hidden_sizes,
        activation=args.activation,
        weight_init=args.weight_init,
        weight_decay=args.weight_decay
    )

    loss_fn = get_loss(args.loss)
    optimiser = get_optimizer(args.optimizer, args.learning_rate)

    model = train(model, X_train, y_train, X_val, y_val,
                  loss_fn, optimiser, args.epochs, args.batch_size, wandb_run)
    
    weights = model.get_weights()
    np.save("src/best_model.npy", weights)
    with open("src/best_config.json", "w") as f:
        json.dump(vars(args), f)

    wandb.finish()

if __name__ == "__main__":
    main()
    
