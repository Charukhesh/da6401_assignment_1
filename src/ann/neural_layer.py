import numpy as np

class Dense:
    def __init__(self, in_features, out_features, weight_init="xavier", weight_decay=0.0):
        self.in_features = in_features
        self.out_features = out_features
        self.weight_decay = weight_decay

        # Weight Initialisation
        if weight_init == "xavier":
            limit = np.sqrt(6 / (in_features + out_features))
            self.W = np.random.uniform(-limit, limit, (in_features, out_features))
        elif weight_init == "random":
            self.W = 0.01 * np.random.randn(in_features, out_features)
        else:
            raise ValueError("Unsupported weight initialisation")
        
        self.b = np.zeros((1, out_features))

        # Gradients 
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

        self.X = None

    def forward(self, X):
        """
        X: (batch_size, in_features)
        Returns: (batch_size, out_features)
        """
        self.X = X
        return X @ self.W + self.b
    
    def backward(self, dZ):
        """
        dZ: (batch_size, out_features)
        Returns: Gradient wrt input X
        """
        batch_size = self.X.shape[0]

        # Gradients
        self.grad_W = (self.X.T @ dZ) / batch_size + self.weight_decay * self.W
        self.grad_b = np.sum(dZ, axis=0, keepdims=True) / batch_size

        # Gradient wrt input
        dX = dZ @ self.W.T
        return dX

