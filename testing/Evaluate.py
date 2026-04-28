import numpy as np

def evaluate(X, y, model):
    logits = model.forward(X)
    preds = np.argmax(logits, axis=1)
    acc = np.mean(preds == y)
    return acc