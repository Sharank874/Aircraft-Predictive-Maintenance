"""
Runs (1) classical baselines and (2) the deep-learning ablation:
    CNN-only, BiLSTM-only, CNN+BiLSTM (mean pooling), CNN+BiLSTM+Attention (full)
each with 3 random seeds, identical unit-level splits and identical evaluation.
Writes results/experiments.json and model weights to models/.
"""
import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from data import prepare
from metrics import best_f1_threshold, fault_metrics, risk_metrics, nasa_score, RUL_CAP
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "results").mkdir(exist_ok=True); (ROOT / "models").mkdir(exist_ok=True)
SEEDS = [0, 1, 2]
VARIANTS = ["cnn", "bilstm", "cnn_bilstm", "full"]


def window_features(X):
    """Hand-engineered window summary for classical baselines: last value, mean, std, slope."""
    t = np.arange(X.shape[1]) - (X.shape[1] - 1) / 2
    slope = (X * t[None, :, None]).sum(1) / (t ** 2).sum()
    return np.concatenate([X[:, -1], X.mean(1), X.std(1), slope], axis=1)


def evaluate(pf_val, yf_val, pf, pr, split, last=None):
    thr = best_f1_threshold(yf_val, pf_val)
    X, yf, yr, rul = split[:4]
    res = {"fault": fault_metrics(yf, pf, thr), "risk": risk_metrics(yr, pr, rul)}
    if last is not None:          # standard C-MAPSS benchmark: last window per test engine
        pf_l, pr_l, (_, yf_l, yr_l, rul_l) = last[0], last[1], last[2][:4]
        rp = (1 - pr_l) * RUL_CAP
        res["benchmark_last_window"] = {**risk_metrics(yr_l, pr_l, rul_l),
                                       "nasa_score": nasa_score(np.minimum(rul_l, RUL_CAP), rp)}
    return res


def run_baselines(d):
    out = {}
    Ftr, Fva, Fte, Fl = (window_features(d[k][0]) for k in ["train", "val", "test", "test_last"])
    _, yf, yr, _ = d["train"][:4]

    # 1) Threshold monitoring on a single health indicator (sensor most correlated with RUL in
    #    TRAINING data), smoothed over the last 5 cycles. Threshold chosen on validation.
    feats = d["feats"]
    corr = [abs(np.corrcoef(d["train"][0][:, -1, j], d["train"][3])[0, 1]) for j in range(len(feats))]
    j = int(np.argmax(corr)); sign = np.sign(np.corrcoef(d["train"][0][:, -1, j], yr)[0, 1])
    hi = lambda X: sign * X[:, -5:, j].mean(1)
    # map indicator to [0,1] with a rank transform fitted on train so ROC/PR are defined
    ref = np.sort(hi(d["train"][0]))
    to_p = lambda X: np.searchsorted(ref, hi(X)) / len(ref)
    thr = best_f1_threshold(d["val"][1], to_p(d["val"][0]))
    out["threshold_monitor"] = {"fault": fault_metrics(d["test"][1], to_p(d["test"][0]), thr),
                                "indicator": feats[j]}

    # 2) Logistic regression (fault) + Ridge (risk) on engineered window features
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(Ftr, yf)
    rg = Ridge(alpha=1.0).fit(Ftr, yr)
    clip = lambda v: np.clip(v, 0, 1)
    out["linear"] = evaluate(lr.predict_proba(Fva)[:, 1], d["val"][1], lr.predict_proba(Fte)[:, 1],
                             clip(rg.predict(Fte)), d["test"],
                             (None, clip(rg.predict(Fl)), d["test_last"]))

    # 3) Random forest on engineered window features
    rfc = RandomForestClassifier(300, min_samples_leaf=5, n_jobs=-1, random_state=0).fit(Ftr, yf)
    rfr = RandomForestRegressor(300, min_samples_leaf=5, n_jobs=-1, random_state=0).fit(Ftr, yr)
    out["random_forest"] = evaluate(rfc.predict_proba(Fva)[:, 1], d["val"][1],
                                    rfc.predict_proba(Fte)[:, 1], rfr.predict(Fte), d["test"],
                                    (None, rfr.predict(Fl), d["test_last"]))
    return out


def run_deep(d):
    from model import build_model, train, predict_det
    T, F = d["train"][0].shape[1:]
    out = {}
    for v in VARIANTS:
        out[v] = []
        for s in SEEDS:
            t0 = time.time()
            m, _ = build_model(T, F, v)
            hist = train(m, d, seed=s)
            pf_va, _ = predict_det(m, d["val"][0])
            pf, pr = predict_det(m, d["test"][0])
            _, pr_l = predict_det(m, d["test_last"][0])
            r = evaluate(pf_va, d["val"][1], pf, pr, d["test"], (None, pr_l, d["test_last"]))
            r.update(seed=s, epochs=len(hist["loss"]), best_val_loss=float(min(hist["val_loss"])),
                     train_seconds=round(time.time() - t0, 1), params=int(m.count_params()))
            out[v].append(r)
            m.save_weights(ROOT / "models" / f"{v}_seed{s}.weights.h5")
            json.dump(hist, open(ROOT / "models" / f"{v}_seed{s}_history.json", "w"))
            print(f"{v:11s} seed={s} ep={r['epochs']:2d} AUC={r['fault']['roc_auc']:.3f} "
                  f"PR-AUC={r['fault']['pr_auc']:.3f} F1={r['fault']['f1']:.3f} "
                  f"RUL-RMSE(last)={r['benchmark_last_window']['rul_rmse']:.2f} "
                  f"t={r['train_seconds']}s", flush=True)
    return out


if __name__ == "__main__":
    d = prepare()
    res = {"data": {"features": d["feats"], "n_train_units": int(d["n_train_units"]),
                    "val_units": d["val_units"], "rtf_units": d["rtf_units"],
                    **{f"n_windows_{k}": int(len(d[k][0])) for k in ["train", "val", "rtf", "test", "test_last"]},
                    **{f"fault_rate_{k}": float(d[k][1].mean()) for k in ["train", "val", "test", "test_last"]}}}
    res["baselines"] = run_baselines(d)
    for k, v in res["baselines"].items():
        print(k, {m: round(x, 3) for m, x in v["fault"].items() if isinstance(x, float)})
    json.dump(res, open(ROOT / "results" / "experiments.json", "w"), indent=2)
    res["deep"] = run_deep(d)
    json.dump(res, open(ROOT / "results" / "experiments.json", "w"), indent=2)
    print("done")
