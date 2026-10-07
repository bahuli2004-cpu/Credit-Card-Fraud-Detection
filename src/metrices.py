import numpy as np
from sklearn.metrics import average_precision_score

def precision_at_k(y_true, scores, k):
    top = np.argsort(-scores)[:k]
    return np.asarray(y_true)[top].mean()

def recall_at_k(y_true, scores, k):
    y_true = np.asarray(y_true)
    top = np.argsort(-scores)[:k]
    return y_true[top].sum() / y_true.sum()

def evaluate(y_true, scores, budgets=(0.001, 0.005, 0.01)):
    n = len(y_true)
    out = {"PR_AUC": average_precision_score(y_true, scores)}
    for b in budgets:
        k = int(n * b)
        out[f"P@{b:.1%}"] = precision_at_k(y_true, scores, k)
        out[f"R@{b:.1%}"] = recall_at_k(y_true, scores, k)
    return out