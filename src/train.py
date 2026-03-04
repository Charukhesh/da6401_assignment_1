import argparse
import json
import numpy as np

from keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split

from ann.neural_network import NeuralNetwork
from ann.layers import Dense
from ann.activations import ReLU, Sigmoid, Tanh
from ann.loss import CrossEntropyLoss, MeanSquaredError
from ann.optimisers import SGD, Momentum, NAG, RMSProp
from ann.training import train

def load_dataset(name):
    if name == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    elif name == "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError("Unsupported dataset")
    
    X_train = X_train.reshape(len(X_train), -1) / 255.0
    X_test = X_test.reshape(len(X_test), -1) / 255.0

    return X_train, y_train, X_test, y_test

def get_activation(name):
    if name == "relu":
        return ReLU()
    elif name == "sigmoid":
        return Sigmoid()
    elif name == "tanh":
        return Tanh()
    else:
        raise ValueError("Unsupported activation")
    
def get_optimiser(name, lr):
    if name == "sgd":
        return SGD(lr)
    elif name == "momentum":
        return Momentum(lr)
    elif name == "nag":
        return NAG(lr)
    elif name == "rmsprop":
        return RMSProp(lr)
    else:
        raise ValueError("Unsupported optimizer")
    
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

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="mnist")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--loss", default="cross_entropy")
    parser.add_argument("--optimiser", default="sgd")
    parser.add_argument("--learning_rate", type=float, default=0.01)

    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_size", nargs="+", type=int, default=[128, 64])

    parser.add_argument("--activation", default="relu")
    parser.add_argument("--weight_init", default="xavier")

    parser.add_argument("--wandb_project", default="ae22b028-da6401-as1")

    args = parser.parse_args()

    X_train, y_train, X_test, y_test = load_dataset(args.dataset)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)

    hidden_sizes = args.hidden_size[: args.num_layers]

    model = build_model(
        input_dim=784,
        hidden_sizes=hidden_sizes,
        activation=args.activation,
        weight_init=args.weight_init,
        weight_decay=args.weight_decay
    )

    loss_fn = get_loss(args.loss)
    optimiser = get_optimiser(args.optimiser, args.learning_rate)

    model = train(model, X_train, y_train, X_val, y_val,
                  loss_fn, optimiser, args.epochs, args.batch_size)
    
    weights = model.get_weights()
    np.save("best_model.npy", weights)
    with open("best_config.json", "w") as f:
        json.dump(vars(args), f)

if __name__ == "__main__":
    main()
    
