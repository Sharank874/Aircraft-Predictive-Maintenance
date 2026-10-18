"""
Builds dashboard.html: a single self-contained, offline, interactive dashboard of the latest run.

Reads   results/experiments.json, results/analysis.json,
        results/engine_trajectories.csv, results/attention_profile.csv
Writes  dashboard.html  (open it in any browser; d3 and all data are embedded)

Runs automatically at the end of src/analyze.py. Run it on its own with:
    python src/make_dashboard.py
"""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
RUL_CAP = 125

MODEL_NAMES = {"threshold_monitor": "Threshold on sensor s11", "linear": "Logistic / Ridge",
               "random_forest": "Random forest", "cnn": "CNN only", "bilstm": "BiLSTM only",
               "cnn_bilstm": "CNN + BiLSTM", "full": "CNN + BiLSTM + Attention"}
CATS = ["Low", "Moderate", "High", "Critical"]


def _r(x, n=4):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), n)


def _models(exp):
    rows = []
    for k, v in exp.get("baselines", {}).items():
        b = v.get("benchmark_last_window", {})
        rows.append({"model": MODEL_NAMES.get(k, k), "family": "Classical baseline", "seeds": 1,
                     "pr_auc": _r(v["fault"]["pr_auc"]), "pr_auc_sd": 0, "f1": _r(v["fault"]["f1"]),
                     "recall": _r(v["fault"]["recall"]), "precision": _r(v["fault"]["precision"]),
                     "rul_rmse_last": _r(b.get("rul_rmse"), 2), "rul_rmse_last_sd": 0,
                     "rul_rmse_all": _r(v.get("risk", {}).get("rul_rmse"), 2)})
    for k, runs in exp.get("deep", {}).items():
        g = lambda f: np.array([f(r) for r in runs], dtype=float)
        pr, rl = g(lambda r: r["fault"]["pr_auc"]), g(lambda r: r["benchmark_last_window"]["rul_rmse"])
        rows.append({"model": MODEL_NAMES.get(k, k), "family": "Deep-learning ablation", "seeds": len(runs),
                     "pr_auc": _r(pr.mean()), "pr_auc_sd": _r(pr.std()), "f1": _r(g(lambda r: r["fault"]["f1"]).mean()),
                     "recall": _r(g(lambda r: r["fault"]["recall"]).mean()),
                     "precision": _r(g(lambda r: r["fault"]["precision"]).mean()),
                     "rul_rmse_last": _r(rl.mean(), 2), "rul_rmse_last_sd": _r(rl.std(), 2),
                     "rul_rmse_all": _r(g(lambda r: r["risk"]["rul_rmse"]).mean(), 2)})
    return rows


def _headline(a):
    det = a["deterministic_vs_mc"]["deterministic"]
    th = a.get("thresholds", {})
    alarm = th.get("alarm_fault_prob", a["shift_fd003"]["fd003_fault"]["threshold"])
    review = th.get("review_risk_std", a["risk_categories"]["review_flag_std_threshold"])
    return [
        {"metric": "pr_auc", "label": "Fault within 30 cycles, PR-AUC", "value": _r(det["fault"]["pr_auc"])},
        {"metric": "precision", "label": "Alarm precision", "value": _r(det["fault"]["precision"])},
        {"metric": "recall", "label": "Alarm recall", "value": _r(det["fault"]["recall"])},
        {"metric": "rul_rmse", "label": "RUL error, RMSE (cycles)", "value": _r(det["risk"]["rul_rmse"], 2)},
        {"metric": "spearman", "label": "Spearman, uncertainty vs error", "value": _r(a["uncertainty"]["spearman_std_vs_abs_error"], 3)},
        {"metric": "coverage95", "label": "Coverage of nominal 95% interval", "value": _r(a["uncertainty"]["nominal95_interval_coverage"])},
        {"metric": "category_accuracy", "label": "Risk category exact accuracy", "value": _r(a["risk_categories"]["exact_category_accuracy"])},
        {"metric": "alarm_threshold", "label": "Alarm threshold on fault probability", "value": _r(alarm)},
        {"metric": "review_std", "label": "Review flag: risk spread above", "value": _r(review)},
    ]


def _records(df):
    return json.loads(df.to_json(orient="records"))


def build(root: Path = ROOT, out: Path | None = None) -> Path:
    res = root / "results"
    exp = json.load(open(res / "experiments.json"))
    a = json.load(open(res / "analysis.json"))
    u, s = a["uncertainty"], a["shift_fd003"]
    cm = a["risk_categories"]["confusion_true_rows_pred_cols"]
    mc = a["deterministic_vs_mc"]["mc_mean"]

    datasets = {
        "models": _models(exp),
        "headline": _headline(a),
        "quintiles": [{"quintile": f"Q{q['quintile']}", "mean_spread_cycles": _r(q["mean_std"] * RUL_CAP, 2),
                       "mae_cycles": _r(q["mae_rul_cycles"], 2)} for q in u["by_std_quintile"]],
        "referral": [{"referred_pct": r["referred_pct"], "by_uncertainty": _r(r["rul_rmse_remaining"], 2),
                      "at_random": _r(r["rul_rmse_random_referral"], 2)} for r in u["referral"]],
        "shift": [{"test_set": "FD001 (seen fault mode)", "rul_rmse": _r(s["fd001_test_rul_rmse"], 2),
                   "mean_spread_cycles": _r(s["fd001_test_mean_risk_std"] * RUL_CAP, 2), "fault_recall": _r(mc["fault"]["recall"])},
                  {"test_set": "FD003 (unseen fan fault)", "rul_rmse": _r(s["fd003_test_rul_rmse"], 2),
                   "mean_spread_cycles": _r(s["fd003_test_mean_risk_std"] * RUL_CAP, 2), "fault_recall": _r(s["fd003_fault"]["recall"])}],
        "categories": [{"true_category": CATS[i], "predicted_category": CATS[j], "windows": int(cm[i][j])}
                       for i in range(4) for j in range(4)],
        "trajectories": None,
        "attention": None,
    }
    tp, ap = res / "engine_trajectories.csv", res / "attention_profile.csv"
    if tp.exists():
        datasets["trajectories"] = _records(pd.read_csv(tp).round(4))
    else:
        print("note: results/engine_trajectories.csv not found; re-run src/analyze.py for the engine explorer")
    if ap.exists():
        datasets["attention"] = _records(pd.read_csv(ap).round(5))

    engines = sorted({r["engine"] for r in datasets["trajectories"]}) if datasets["trajectories"] else []
    stamp = dt.datetime.fromtimestamp((res / "analysis.json").stat().st_mtime).strftime("%d %b %Y, %H:%M")
    payload = {"datasets": datasets, "params": {"engine": engines[0] if engines else None},
               "meta": {"built": dt.datetime.now().strftime("%d %b %Y, %H:%M"),
                        "source": f"the run analysed on {stamp} (selected seed {a.get('selected_seed')}, {a.get('mc_passes', 50)} MC Dropout passes)"}}

    html = (SRC / "dashboard_template.html").read_text(encoding="utf-8")
    d3 = (SRC / "vendor" / "d3.v7.min.js").read_text(encoding="utf-8")
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = html.replace("/*__D3__*/", d3.replace("</script", "<\\/script"), 1).replace("/*__DATA__*/", data, 1)
    out = out or (root / "dashboard.html")
    out.write_text(html, encoding="utf-8")
    return out


if __name__ == "__main__":
    print("Dashboard written to", build())
