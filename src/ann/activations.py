import numpy as np

class ReLU:
    def __init__(self):
        self.Z = None

    def forward(self, X):
        self.Z = X
        return np.maximum(0, X)

    def backward(self, dA):
        dZ = dA.copy()
        dZ[self.Z <= 0] = 0
        return dZ
    
class Sigmoid:
    def __init__(self):
        self.A = None

    def forward(self, X):
        self.A = 1 / (1 + np.exp(-X))
        return self.A
    
    def backward(self, dA):
        return dA * self.A * (1 - self.A)
    
class Tanh:
    def __init__(self):
        self.A = None

    def forward(self, X):
        self.A = np.tanh(X)
        return self.A
    
    def backward(self, dA):
        return dA * (1 - self.A**2)