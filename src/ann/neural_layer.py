import numpy as np

class Dense:

    def __init__(self, in_features, out_features, weight_init="xavier", weight_decay=0.0):

        self.weight_decay = weight_decay

        if weight_init == "xavier":
            limit = np.sqrt(6/(in_features + out_features))
            self.W = np.random.uniform(-limit, limit, (in_features, out_features))
        else:
            self.W = 0.01*np.random.randn(in_features, out_features)

        self.b = np.zeros((1,out_features))

        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

    def forward(self,X):
        self.X = X
        return X @ self.W + self.b

    def backward(self,dZ):

        batch_size = self.X.shape[0]

        self.grad_W = self.X.T @ dZ
        self.grad_b = np.sum(dZ, axis=0, keepdims=True)

        if self.weight_decay > 0:
            self.grad_W += self.weight_decay * self.W

        return dZ @ self.W.T