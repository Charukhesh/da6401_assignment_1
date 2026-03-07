# Tensorflow in keras enables CPU acceleration, suppressing the warning
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from keras.datasets import mnist, fashion_mnist

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