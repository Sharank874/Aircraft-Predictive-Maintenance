import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_fscore_support,
                             mean_absolute_error, mean_squared_error, r2_score, brier_score_loss,
                             precision_recall_curve)

RUL_CAP = 125


def best_f1_threshold(y, p):
    """Pick the decision threshold that maximises F1 -- called on VALIDATION data only."""
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / np.clip(pr + rc, 1e-9, None)
    i = int(np.nanargmax(f1[:-1]))
    return float(th[i])


def ece(y, p, bins=10):
    """Expected calibration error of predicted fault probabilities."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def fault_metrics(y, p, thr):
    yhat = (p >= thr).astype(int)
    P, R, F, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "precision": float(P), "recall": float(R), "f1": float(F), "threshold": thr,
            "brier": float(brier_score_loss(y, p)), "ece": ece(y, p),
            "positive_rate": float(y.mean())}


def nasa_score(rul_true, rul_pred):
    d = rul_pred - rul_true          # late predictions (d > 0) are penalised harder
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def risk_metrics(r_true, r_pred, rul_true=None):
    out = {"risk_mae": float(mean_absolute_error(r_true, r_pred)),
           "risk_rmse": float(np.sqrt(mean_squared_error(r_true, r_pred))),
           "risk_r2": float(r2_score(r_true, r_pred))}
    if rul_true is not None:       # translate the risk index back into cycles of RUL
        rt = np.minimum(rul_true, RUL_CAP)
        rp = (1 - r_pred) * RUL_CAP
        out["rul_rmse"] = float(np.sqrt(mean_squared_error(rt, rp)))
        out["rul_mae"] = float(mean_absolute_error(rt, rp))
    return out
