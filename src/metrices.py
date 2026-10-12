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
def dollars_caught_at_k(y_true, scores, amount, k):
    y_true, amount = np.asarray(y_true), np.asarray(amount)
    top = np.argsort(-scores)[:k]
    return (amount[top] * y_true[top]).sum() / (amount * y_true).sum()

def bootstrap_ci(y_true, scores, metric_fn, n_boot=300, seed=42, **kw):
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    r = np.random.default_rng(seed)
    n, vals = len(y_true), []
    for _ in range(n_boot):
        idx = r.integers(0, n, n)          # resample rows with replacement
        vals.append(metric_fn(y_true[idx], scores[idx], **kw))
    return np.percentile(vals, [2.5, 97.5])