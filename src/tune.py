"""
Validation-only tuning of the regularisation recipe for the full model.
v1 (no augmentation, 64 units) overfit within ~8 epochs (train risk MSE kept falling, validation
flattened). This sweep tries a few regularisation settings and picks one by VALIDATION
RUL RMSE averaged over 2 seeds. The NASA test set is never touched here.
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from data import prepare
from model import build_model, train, predict_det
from metrics import RUL_CAP
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "v1_baseline":        dict(),
    "noise0.1":           dict(noise=0.1),
    "noise0.1_l2":        dict(noise=0.1, l2=1e-4),
    "small32_noise0.1":   dict(noise=0.1, conv_filters=32, lstm_units=32, dense_units=32),
    "noise0.2_lr5e-4":    dict(noise=0.2, lr=5e-4),
}

if __name__ == "__main__":
    d = prepare()
    T, F = d["train"][0].shape[1:]
    Xv, vf, vr, vrul = d["val"][:4]
    out = {}
    for name, cfg in CONFIGS.items():
        rm, pa = [], []
        for s in [0, 1]:
            m, _ = build_model(T, F, "full", **cfg)
            train(m, d, seed=s, patience=10)
            pf, pr = predict_det(m, Xv)
            rm.append(float(np.sqrt(np.mean(((pr - vr) * RUL_CAP) ** 2))))
            pa.append(float(average_precision_score(vf, pf)))
        out[name] = {"cfg": cfg, "val_rul_rmse": rm, "val_pr_auc": pa,
                     "mean_val_rul_rmse": float(np.mean(rm)), "mean_val_pr_auc": float(np.mean(pa))}
        print(f"{name:18s} val RUL-RMSE={np.mean(rm):.2f} {rm}  val PR-AUC={np.mean(pa):.3f}", flush=True)
    best = min(out, key=lambda k: out[k]["mean_val_rul_rmse"])
    out["selected"] = best
    json.dump(out, open(ROOT / "results" / "tuning_validation.json", "w"), indent=2)
    print("selected:", best)
