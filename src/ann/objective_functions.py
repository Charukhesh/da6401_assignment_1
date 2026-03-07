import numpy as np

class MeanSquaredError:
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred, y_true):
        """
        y_pred: (batch_size, num_classes)
        y_true: (batch_size, num_classes)
        """
        self.y_pred = y_pred
        self.y_true = y_true

        loss = np.mean((y_true - y_pred)**2)
        return loss
    
    def backward(self):
        """
        Returns gradient wrt y_pred
        """
        batch_size = self.y_true.shape[0]
        return (2 / batch_size) * (self.y_pred - self.y_true)
    
class CrossEntropyLoss:
    def __init__(self):
        self.probs = None
        self.y_true = None

    def forward(self, logits, y_true):
        """
        logits: (batch_size, num_classes)
        y_true: one-hot encoded (batch_size, num_classes)
        """
        self.y_true = y_true

        # To prevent overflow, we use softmax's invariance to constant shifts property
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_vals = np.exp(shifted_logits)
        self.probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

        log_probs = np.log(self.probs + 1e-12)
        loss = -np.sum(y_true * log_probs) / logits.shape[0]

        return loss
    
    def backward(self):
        """
        Gradient wrt logits
        """
        batch_sze = self.y_true.shape[0]
        return (self.probs - self.y_true) / batch_sze