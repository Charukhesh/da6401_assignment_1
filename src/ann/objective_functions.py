import numpy as np

class MeanSquaredError:
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_true, logits):
        """
        logits: (batch_size, num_classes)
        y_true: (batch_size, num_classes)
        """
        self.y_pred = logits
        self.y_true = y_true

        loss = np.sum((y_true - logits)**2) / logits.shape[0]
        return loss
    
    def backward(self, y_true, logits):
        """
        Returns gradient wrt y_pred
        """
        if y_true.ndim == 1:
            y_true = np.eye(logits.shape[1])[y_true]

        batch_size = y_true.shape[0]
        return (2 / batch_size) * (logits - y_true)
    
class CrossEntropyLoss:
    def __init__(self):
        self.probs = None
        self.y_true = None

    def forward(self, y_true, logits):
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
    
    def backward(self, y_true, logits):
        """
        Gradient wrt logits
        """

        if y_true.ndim == 1:
            y_true = np.eye(logits.shape[1])[y_true]

        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

        batch_size = y_true.shape[0]

        return (probs - y_true) / batch_size