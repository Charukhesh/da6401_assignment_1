import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        """
        Add layer to model
        """
        self.layers.append(layer)

    def forward(self, X):
        """
        Forward pass through all layers.
        Returns logits.
        """
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def backward(self, grad):
        """
        Backward pass through all layers:
        grad: Gradient wrt logits
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def get_weights(self):
        """
        Returns dictionary of weights for saving.
        """
        weights = {}
        for idx, layer in enumerate(self.layers):
            if hasattr(layer, "W"):
                weights[f"layer_{idx}_W"] = layer.W
                weights[f"layer_{idx}_b"] = layer.b
        
        return weights
    
    def set_weights(self, weights_dict):
        """
        Load weights from dictionary.
        """
        for idx, layer in enumerate(self.layers):
            if hasattr(layer, "W"):
                layer.W = weights_dict[f"layer_{idx}_W"]
                layer.b = weights_dict[f"layer_{idx}_b"]