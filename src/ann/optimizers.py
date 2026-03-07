import numpy as np

class SGD:
    def __init__(self, learning_rate):
        self.lr = learning_rate

    def update(self, layers):
        for layer in layers:
            if hasattr(layer, "W"):
                layer.W -= self.lr * layer.grad_W
                layer.b -= self.lr * layer.grad_b

class Momentum:
    def __init__(self, learning_rate, beta=0.9):
        self.lr = learning_rate
        self.beta = beta
        self.velocities = {}

    def update(self, layers):
        for idx, layer in enumerate(layers):
            if hasattr(layer, "W"):
                if idx not in self.velocities:
                    self.velocities[idx] = {
                        "vW": np.zeros_like(layer.W),
                        "vB": np.zeros_like(layer.b),
                    }

                self.velocities[idx]["vW"] = (
                    self.beta * self.velocities[idx]["vW"]
                    + self.lr * layer.grad_W
                )

                self.velocities[idx]["vB"] = (
                    self.beta * self.velocities[idx]["vB"]
                    + self.lr * layer.grad_b
                )

                layer.W -= self.velocities[idx]["vW"]
                layer.b -= self.velocities[idx]["vB"]

class NAG:
    def __init__(self, learning_rate, beta=0.9):
        self.lr = learning_rate
        self.beta = beta
        self.velocities = {}

    def update(self, layers):
        for idx, layer in enumerate(layers):
            if hasattr(layer, "W"):
                if idx not in self.velocities:
                    self.velocities[idx] = {
                        "vW": np.zeros_like(layer.W),
                        "vB": np.zeros_like(layer.b),
                    }

                v_prev_W = self.velocities[idx]["vW"]
                v_prev_B = self.velocities[idx]["vB"]

                self.velocities[idx]["vW"] = (
                    self.beta * v_prev_W + self.lr * layer.grad_W
                )
                self.velocities[idx]["vB"] = (
                    self.beta * v_prev_B + self.lr * layer.grad_b
                )

                layer.W -= (
                    self.beta * v_prev_W + self.lr * layer.grad_W
                )
                layer.b -= (
                    self.beta * v_prev_B + self.lr * layer.grad_b
                )

class RMSProp:
    def __init__(self, learning_rate, beta=0.9, epsilon=1e-8):
        self.lr = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.cache = {}

    def update(self, layers):
        for idx, layer in enumerate(layers):
            if hasattr(layer, "W"):
                if idx not in self.cache:
                    self.cache[idx] = {
                        "sW": np.zeros_like(layer.W),
                        "sB": np.zeros_like(layer.b),
                    }

                self.cache[idx]["sW"] = (
                    self.beta * self.cache[idx]["sW"] 
                    + (1 - self.beta) * (layer.grad_W**2)
                )
                self.cache[idx]["sB"] = (
                    self.beta * self.cache[idx]["sB"] 
                    + (1 - self.beta) * (layer.grad_b**2)
                )

                layer.W -= (
                    self.lr * layer.grad_W / (np.sqrt(self.cache[idx]["sW"] + self.epsilon))
                )
                layer.b -= (
                    self.lr * layer.grad_b / (np.sqrt(self.cache[idx]["sB"] + self.epsilon))
                )

    