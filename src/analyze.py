"""
Uncertainty, risk-scoring, attention, lead-time, drift and latency analysis for the full
CNN+BiLSTM+Attention model, plus a simulated streaming scorer and the report figures.
Writes results/analysis.json, results/streaming_demo.csv, results/engine_trajectories.csv,
results/attention_profile.csv, figures/*.png, and rebuilds dashboard.html at the end.
"""
import json, sys, time, warnings
warnings.filterwarnings("ignore", message="Skipping variable loading for optimizer")
from collections import deque
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from data import prepare, WINDOW, RUL_CAP, HORIZON
from model import build_model, predict_det, predict_mc, set_mc, keras
from metrics import fault_metrics, risk_metrics, best_f1_threshold, nasa_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
M = 50  # Monte Carlo passes
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
C1, C2, C3, C4 = "#2f6db5", "#d1495b", "#3a9d6a", "#8a8a8a"

# Risk categories expressed on the risk index (risk = 1 - min(RUL,125)/125):
#   Critical: predicted RUL <= 15 cycles   High: 15-30   Moderate: 30-60   Low: > 60
CAT_EDGES = [1 - 60 / RUL_CAP, 1 - 30 / RUL_CAP, 1 - 15 / RUL_CAP]   # 0.52, 0.76, 0.88
CATS = ["Low", "Moderate", "High", "Critical"]
categorize = lambda r: np.digitize(r, CAT_EDGES)


def main():
    # Seed numpy, TF and Keras BEFORE the model (and its dropout layers) is built, so the
    # Monte Carlo Dropout masks -- and therefore every uncertainty number -- are reproducible.
    keras.utils.set_random_seed(0)
    exp = json.load(open(ROOT / "results" / "experiments.json"))
    d = prepare(test_subset="FD003")
    T, F = d["train"][0].shape[1:]

    # Pick the seed with the best VALIDATION loss (never by test score)
    runs = exp["deep"]["full"]
    seed = min(runs, key=lambda r: r["best_val_loss"])["seed"]
    model, att_model = build_model(T, F, "full")
    model.load_weights(ROOT / "models" / f"full_seed{seed}.weights.h5")
    out = {"selected_seed": seed, "mc_passes": M, "dropout": 0.3,
           "dropout_layers_active_at_inference": ["after Conv1D block", "after BiLSTM", "after Dense(64)"]}

    import tensorflow as tf
    mc = {k: predict_mc(model, d[k][0], passes=M) for k in ["val", "test", "test_last", "rtf", "shift"]}
    det = {k: predict_det(model, d[k][0]) for k in ["val", "test"]}

    # ---------- 1. MC-mean vs deterministic ----------
    thr_det = best_f1_threshold(d["val"][1], det["val"][0])
    thr_mc = best_f1_threshold(d["val"][1], mc["val"]["fault_mean"])
    Xt, yf, yr, rul = d["test"][:4]
    out["deterministic_vs_mc"] = {
        "deterministic": {"fault": fault_metrics(yf, det["test"][0], thr_det),
                          "risk": risk_metrics(yr, det["test"][1], rul)},
        "mc_mean": {"fault": fault_metrics(yf, mc["test"]["fault_mean"], thr_mc),
                    "risk": risk_metrics(yr, mc["test"]["risk_mean"], rul)}}
    _, _, yr_l, rul_l = d["test_last"][:4]
    out["benchmark_last_window_mc"] = {**risk_metrics(yr_l, mc["test_last"]["risk_mean"], rul_l),
        "nasa_score": nasa_score(np.minimum(rul_l, RUL_CAP), (1 - mc["test_last"]["risk_mean"]) * RUL_CAP)}

    # ---------- 2. Does uncertainty track error? ----------
    rm, rs = mc["test"]["risk_mean"], mc["test"]["risk_std"]
    err = np.abs(rm - yr)
    rho, pval = spearmanr(rs, err)
    q = pd.qcut(rs, 5, labels=False)
    quint = [{"quintile": int(i + 1), "mean_std": float(rs[q == i].mean()),
              "mae_risk": float(err[q == i].mean()),
              "mae_rul_cycles": float((err[q == i] * RUL_CAP).mean())} for i in range(5)]
    # referral: hand the k% most-uncertain windows to an engineer, score the rest
    order = np.argsort(-rs); rng = np.random.default_rng(0)
    referral = []
    for k in [0, 5, 10, 20, 30]:
        keep = np.ones(len(rs), bool); keep[order[:int(len(rs) * k / 100)]] = False
        rnd = np.ones(len(rs), bool); rnd[rng.choice(len(rs), int(len(rs) * k / 100), replace=False)] = False
        referral.append({"referred_pct": k, "rul_rmse_remaining": float(np.sqrt(np.mean(((rm - yr)[keep] * RUL_CAP) ** 2))),
                         "rul_rmse_random_referral": float(np.sqrt(np.mean(((rm - yr)[rnd] * RUL_CAP) ** 2)))})
    # interval coverage: does mean +/- 1.96*std contain the truth 95% of the time?
    cover = float(np.mean(np.abs(rm - yr) <= 1.96 * rs))
    # fault head: uncertainty on wrong vs right decisions
    fm, fs = mc["test"]["fault_mean"], mc["test"]["fault_std"]
    wrong = (fm >= thr_mc).astype(int) != yf
    out["uncertainty"] = {"spearman_std_vs_abs_error": float(rho), "p_value": float(pval),
                          "by_std_quintile": quint, "referral": referral,
                          "nominal95_interval_coverage": cover,
                          "fault_std_correct": float(fs[~wrong].mean()), "fault_std_wrong": float(fs[wrong].mean()),
                          "n_wrong": int(wrong.sum())}

    # ---------- 3. Distribution shift: FD001-trained model on FD003 (extra fan-fault mode) ----------
    Xs, yfs, yrs, ruls = d["shift"][:4]
    out["shift_fd003"] = {
        "fd001_test_mean_risk_std": float(rs.mean()), "fd003_test_mean_risk_std": float(mc["shift"]["risk_std"].mean()),
        "fd001_test_rul_rmse": float(np.sqrt(np.mean(((rm - yr) * RUL_CAP) ** 2))),
        "fd003_test_rul_rmse": float(np.sqrt(np.mean(((mc["shift"]["risk_mean"] - yrs) * RUL_CAP) ** 2))),
        "fd001_std_near_failure": float(rs[rul <= 60].mean()),
        "fd003_std_near_failure": float(mc["shift"]["risk_std"][ruls <= 60].mean()),
        "fd003_fault": fault_metrics(yfs, mc["shift"]["fault_mean"], thr_mc)}

    # ---------- 4. Risk categories + review flag ----------
    std_flag = float(np.quantile(mc["val"]["risk_std"], 0.90))   # top-10% val uncertainty -> review
    pc, tc = categorize(rm), categorize(yr)
    conf = pd.crosstab(pd.Series(tc, name="true"), pd.Series(pc, name="pred")).reindex(index=range(4), columns=range(4), fill_value=0)
    out["risk_categories"] = {"edges_on_risk_index": CAT_EDGES, "labels": CATS,
                              "confusion_true_rows_pred_cols": conf.values.tolist(),
                              "exact_category_accuracy": float((pc == tc).mean()),
                              "within_one_category": float((np.abs(pc - tc) <= 1).mean()),
                              "review_flag_std_threshold": std_flag,
                              "review_flag_rate_test": float((rs > std_flag).mean())}

    # ---------- 5. Lead time on held-out run-to-failure engines ----------
    Xr, yfr, yrr, rulr, ur, cr = d["rtf"]
    pr_f = mc["rtf"]["fault_mean"]
    lead = []
    for u in np.unique(ur):
        m = ur == u
        a = pr_f[m] >= thr_mc
        run = np.convolve(a.astype(int), np.ones(3, int), "valid") >= 3   # 3 consecutive cycles
        idx = np.argmax(run) + 2 if run.any() else None
        lead.append({"unit": int(u), "life_cycles": int(cr[m].max()),
                     "alarm_rul": None if idx is None else float(rulr[m][idx])})
    leads = np.array([l["alarm_rul"] for l in lead if l["alarm_rul"] is not None])
    out["lead_time_rtf"] = {"alarm_rule": f"MC-mean fault prob >= {thr_mc:.2f} for 3 consecutive cycles",
                            "engines": len(lead), "detected": int(len(leads)),
                            "median_lead_cycles": float(np.median(leads)), "min_lead": float(leads.min()),
                            "max_lead": float(leads.max()),
                            "early_alarms_rul_gt_60": int((leads > 60).sum()), "late_alarms_rul_lt_10": int((leads < 10).sum()),
                            "per_engine": lead}

    # ---------- 6. Attention ----------
    A = att_model.predict(Xt, batch_size=2048, verbose=0)
    ent = -(A * np.log(A + 1e-12)).sum(1)
    out["attention"] = {"mean_weight_last5_near_failure": float(A[yf == 1][:, -5:].sum(1).mean()),
                        "mean_weight_last5_healthy": float(A[rul > 100][:, -5:].sum(1).mean()),
                        "uniform_last5": 5 / WINDOW, "mean_entropy": float(ent.mean()),
                        "max_entropy_uniform": float(np.log(WINDOW))}

    # ---------- 7. Latency + streaming ----------
    x1 = Xt[:1]
    for _ in range(3): model(x1, training=False)
    t = time.perf_counter(); [model(x1, training=False) for _ in range(50)]; lat1 = (time.perf_counter() - t) / 50
    xr = np.repeat(x1, M, 0); set_mc(model, True); model(xr)
    t = time.perf_counter(); [model(xr) for _ in range(20)]; latmc = (time.perf_counter() - t) / 20; set_mc(model, False)
    t = time.perf_counter(); predict_mc(model, Xt, passes=M); bulk = time.perf_counter() - t
    # same MC-50 call compiled into a TensorFlow graph (what a deployed service would run)
    set_mc(model, True)
    fast = tf.function(lambda x: model(x, training=False), reduce_retracing=True)
    xr_t = tf.constant(xr); fast(xr_t); fast(xr_t)
    t = time.perf_counter(); [fast(xr_t) for _ in range(50)]; latc = (time.perf_counter() - t) / 50
    set_mc(model, False)
    out["latency_cpu_2core"] = {"single_window_deterministic_eager_ms": lat1 * 1e3,
                                "single_window_mc50_batched_eager_ms": latmc * 1e3,
                                "single_window_mc50_batched_compiled_ms": latc * 1e3,
                                "bulk_mc50_windows_per_second": len(Xt) / bulk}
    stream = streaming_demo(model, d, thr_mc, std_flag)
    out["streaming_demo"] = {"engine": int(stream.unit.iloc[0]), "updates": len(stream),
                             "median_update_ms": float(stream.latency_ms.median()),
                             "p95_update_ms": float(stream.latency_ms.quantile(0.95))}
    stream.to_csv(ROOT / "results" / "streaming_demo.csv", index=False)

    out["thresholds"] = {"alarm_fault_prob": float(thr_mc), "review_risk_std": std_flag,
                         "alarm_rule": "fault probability >= alarm threshold for 3 consecutive cycles"}
    json.dump(out, open(ROOT / "results" / "analysis.json", "w"), indent=2)

    # per-cycle predictions for the held-out run-to-failure engines (dashboard engine explorer)
    traj = pd.DataFrame({"engine": ur.astype(int), "cycle": cr.astype(int), "true_rul": rulr,
                         "true_risk": yrr, "pred_risk": mc["rtf"]["risk_mean"],
                         "risk_std": mc["rtf"]["risk_std"], "fault_prob": mc["rtf"]["fault_mean"]})
    traj["pred_rul"] = (1 - traj.pred_risk) * RUL_CAP
    traj.sort_values(["engine", "cycle"]).round(4).to_csv(ROOT / "results" / "engine_trajectories.csv", index=False)
    # mean attention weight per window position (0 = latest cycle)
    pd.DataFrame({"position": np.arange(-WINDOW + 1, 1), "near_failure": A[yf == 1].mean(0),
                  "healthy": A[rul > 100].mean(0)}).round(5).to_csv(ROOT / "results" / "attention_profile.csv", index=False)

    figures(exp, d, mc, A, out, thr_mc)
    print(json.dumps({k: v for k, v in out.items() if k not in ("lead_time_rtf",)}, indent=1)[:6000])


class StreamingScorer:
    """Per-engine rolling buffer. Each new sensor reading -> one uncertainty-aware risk record.
    Simulates on-line use: only readings up to the current cycle are ever in the buffer."""
    def __init__(self, model, scaler, feats, thr, std_flag, passes=M):
        self.m, self.sc, self.f, self.thr, self.flag, self.M = model, scaler, feats, thr, std_flag, passes
        self.buf = {}

    def update(self, unit, cycle, reading: dict):
        z = ((pd.Series(reading)[self.f] - self.sc.mu[self.f]) / self.sc.sd[self.f]).to_numpy(np.float32)
        b = self.buf.setdefault(unit, deque(maxlen=WINDOW)); b.append(z)
        if len(b) < WINDOW:
            return None                                     # not enough history yet
        x = np.repeat(np.stack(b)[None], self.M, 0)          # M copies in one batch = M MC passes
        set_mc(self.m, True); o = self.m(x, training=False); set_mc(self.m, False)
        f, r = o["fault"].numpy().ravel(), o["risk"].numpy().ravel()
        rmean, rstd = float(r.mean()), float(r.std(ddof=1))
        return {"unit": unit, "cycle": cycle, "fault_prob": float(f.mean()), "fault_alarm": bool(f.mean() >= self.thr),
                "risk": rmean, "risk_std": rstd, "pred_rul": (1 - rmean) * RUL_CAP,
                "category": CATS[int(categorize(np.array([rmean]))[0])],
                "review_flag": rstd > self.flag}


def streaming_demo(model, d, thr, std_flag):
    raw = d["raw_train"]; u = d["rtf_units"][0]
    eng = raw[raw.unit == u]
    sc = StreamingScorer(model, d["scaler"], d["feats"], thr, std_flag)
    rows = []
    for _, r in eng.iterrows():
        t = time.perf_counter(); rec = sc.update(int(u), int(r.cycle), r.to_dict()); dt = (time.perf_counter() - t) * 1e3
        if rec:
            rec.update(true_rul=float(r.rul), latency_ms=dt); rows.append(rec)
    return pd.DataFrame(rows)


def figures(exp, d, mc, A, out, thr):
    raw, feats = d["raw_train"], d["feats"]
    # Fig 1: raw sensor trajectories
    fig, ax = plt.subplots(2, 2, figsize=(8, 4.6))
    for a, s in zip(ax.ravel(), ["s2", "s4", "s11", "s12"]):
        for u in d["rtf_units"][:6]:
            g = raw[raw.unit == u]; a.plot(g.rul, g[s], lw=0.8, alpha=0.8)
        a.invert_xaxis(); a.set_title(f"sensor {s}"); a.set_xlabel("cycles to failure (RUL)")
    fig.suptitle("Raw sensor trajectories of 6 held-out engines drifting toward failure", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "01_sensor_trajectories.png"); plt.close(fig)

    # Fig 2: risk + uncertainty over life for 3 held-out run-to-failure engines
    Xr, yf, yr, rul, ur, cr = d["rtf"]
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for a, u in zip(ax, d["rtf_units"][:3]):
        m = ur == u; o = np.argsort(cr[m])
        c, rm, rs, rt = cr[m][o], mc["rtf"]["risk_mean"][m][o], mc["rtf"]["risk_std"][m][o], yr[m][o]
        a.fill_between(c, rm - 2 * rs, rm + 2 * rs, color=C1, alpha=0.2, label="±2σ (MC Dropout)")
        a.plot(c, rm, color=C1, lw=1.4, label="predicted risk")
        a.plot(c, rt, color="k", lw=1, ls="--", label="true risk index")
        for e in CAT_EDGES: a.axhline(e, color=C4, lw=0.6, ls=":")
        a.set_title(f"engine {u}"); a.set_xlabel("cycle")
    ax[0].set_ylabel("risk index (0 healthy → 1 failure)"); ax[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "02_risk_over_life.png"); plt.close(fig)

    # Fig 3: ablation
    names = {"threshold_monitor": "Threshold\nmonitor", "linear": "Logistic/\nRidge", "random_forest": "Random\nForest",
             "cnn": "CNN", "bilstm": "BiLSTM", "cnn_bilstm": "CNN+\nBiLSTM", "full": "CNN+BiLSTM\n+Attention"}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    labs, pr_m, pr_s, rm_m, rm_s = [], [], [], [], []
    for k in ["threshold_monitor", "linear", "random_forest"]:
        b = exp["baselines"][k]; labs.append(names[k]); pr_m.append(b["fault"]["pr_auc"]); pr_s.append(0)
        rm_m.append(b.get("benchmark_last_window", {}).get("rul_rmse", np.nan)); rm_s.append(0)
    for k in ["cnn", "bilstm", "cnn_bilstm", "full"]:
        rs = exp["deep"][k]; labs.append(names[k])
        v = [r["fault"]["pr_auc"] for r in rs]; pr_m.append(np.mean(v)); pr_s.append(np.std(v))
        v = [r["benchmark_last_window"]["rul_rmse"] for r in rs]; rm_m.append(np.mean(v)); rm_s.append(np.std(v))
    col = [C4] * 3 + [C1] * 3 + [C2]
    ax[0].bar(labs, pr_m, yerr=pr_s, color=col, capsize=3); ax[0].set_title("Fault-within-30-cycles: PR-AUC (higher = better)")
    ax[0].set_ylim(min(pr_m) * 0.9, 1)
    ax[1].bar(labs, rm_m, yerr=rm_s, color=col, capsize=3); ax[1].set_title("RUL RMSE on NASA test engines, cycles (lower = better)")
    for a in ax: a.tick_params(axis="x", labelsize=7)
    fig.tight_layout(); fig.savefig(FIG / "03_baselines_ablation.png"); plt.close(fig)

    # Fig 4: uncertainty vs error + referral
    u = out["uncertainty"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].bar([f"Q{q['quintile']}" for q in u["by_std_quintile"]], [q["mae_rul_cycles"] for q in u["by_std_quintile"]], color=C1)
    ax[0].set_title("Error by MC-Dropout uncertainty quintile"); ax[0].set_ylabel("MAE (cycles of RUL)")
    ax[0].set_xlabel("Q1 = most confident … Q5 = least confident")
    k = [r["referred_pct"] for r in u["referral"]]
    ax[1].plot(k, [r["rul_rmse_remaining"] for r in u["referral"]], "o-", color=C2, label="refer most-uncertain")
    ax[1].plot(k, [r["rul_rmse_random_referral"] for r in u["referral"]], "o--", color=C4, label="refer at random")
    ax[1].set_xlabel("% of windows sent for human review"); ax[1].set_ylabel("RUL RMSE on the rest (cycles)")
    ax[1].set_title("Using uncertainty to triage"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "04_uncertainty_vs_error.png"); plt.close(fig)

    # Fig 5: attention profile
    yf_t, rul_t = d["test"][1], d["test"][3]
    fig, ax = plt.subplots(figsize=(6, 2.8))
    pos = np.arange(-WINDOW + 1, 1)
    ax.plot(pos, A[yf_t == 1].mean(0), color=C2, label="windows with RUL ≤ 30")
    ax.plot(pos, A[rul_t > 100].mean(0), color=C3, label="windows with RUL > 100")
    ax.axhline(1 / WINDOW, color=C4, ls=":", label="uniform (mean pooling)")
    ax.set_xlabel("time step relative to prediction time (0 = latest cycle)"); ax.set_ylabel("mean attention weight")
    ax.legend(fontsize=7); ax.set_title("Where the attention layer looks inside the 30-cycle window")
    fig.tight_layout(); fig.savefig(FIG / "05_attention_profile.png"); plt.close(fig)

    # Fig 6: shift
    fig, ax = plt.subplots(figsize=(6, 2.8))
    bins = np.linspace(0, max(mc["test"]["risk_std"].max(), mc["shift"]["risk_std"].max()), 50)
    ax.hist(mc["test"]["risk_std"] * RUL_CAP, bins=bins * RUL_CAP, alpha=0.6, density=True, color=C1, label="FD001 test (same fault mode)")
    ax.hist(mc["shift"]["risk_std"] * RUL_CAP, bins=bins * RUL_CAP, alpha=0.6, density=True, color=C2, label="FD003 test (unseen fan fault mode)")
    ax.set_xlabel("MC-Dropout std of prediction (cycles of RUL)"); ax.set_ylabel("density"); ax.legend(fontsize=7)
    ax.set_title("Uncertainty under distribution shift")
    fig.tight_layout(); fig.savefig(FIG / "06_shift_uncertainty.png"); plt.close(fig)

    # Fig 7: training curve of selected run
    h = json.load(open(ROOT / "models" / f"full_seed{out['selected_seed']}_history.json"))
    fig, ax = plt.subplots(figsize=(5, 2.6))
    ax.plot(h["loss"], color=C1, label="train"); ax.plot(h["val_loss"], color=C2, label="validation")
    ax.set_xlabel("epoch"); ax.set_ylabel("joint loss"); ax.set_yscale("log"); ax.legend(fontsize=7)
    ax.set_title("Training curve (early stopping on validation engines)")
    fig.tight_layout(); fig.savefig(FIG / "07_training_curve.png"); plt.close(fig)


if __name__ == "__main__":
    main()
    from make_dashboard import build
    print("Dashboard written to", build())
