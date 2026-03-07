import numpy as np
import wandb
from sklearn.metrics import precision_score, recall_score, f1_score

def one_hot_encode(y, num_classes=10):
    """
    Convert label vector to one-hot encoding.
    """
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot

def compute_accuracy(logits, y_true):
    """
    logits: model outputs
    y_true: integer labels
    """
    preds = np.argmax(logits, axis=1)
    return np.mean(preds == y_true)

def create_batches(X, y, batch_size):
    """
    Yield mini-batches
    """
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    for start in range(0, len(X), batch_size):
        end = start + batch_size
        yield X[start:end], y[start:end]

def train(model, X_train, y_train, X_val, y_val,
          loss_fn, optimiser, epochs, batch_size, wandb_run=None):
    """ 
    Training loop
    """
    y_val_oh = one_hot_encode(y_val)

    best_f1 = -1
    best_weights = None

    for epoch in range(epochs):
        total_loss = 0
        total_acc = 0
        batch_count = 0

        for X_batch, y_batch in create_batches(X_train, y_train, batch_size):
            y_batch_oh = one_hot_encode(y_batch)

            # Forward pass
            logits = model.forward(X_batch)

            # Loss
            loss = loss_fn.forward(logits, y_batch_oh)

            # Backward pass
            grad = loss_fn.backward()
            model.backward(grad)

            # Update weights
            optimiser.update(model.layers)

            # Metrics
            acc = compute_accuracy(logits, y_batch)

            total_loss += loss
            total_acc += acc
            batch_count += 1

        train_loss = total_loss / batch_count
        train_acc = total_acc / batch_count

        # Validation
        val_logits = model.forward(X_val)
        val_loss = loss_fn.forward(val_logits, y_val_oh)
        val_acc = compute_accuracy(val_logits, y_val)

        val_preds = np.argmax(val_logits, axis=1)
        val_precision = precision_score(y_val, val_preds, average="macro")
        val_recall = recall_score(y_val, val_preds, average="macro")
        val_f1 = f1_score(y_val, val_preds, average="macro")

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )
        
        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
            })

    return model



















