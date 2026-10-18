# Aircraft Predictive Maintenance & Risk Scoring

Predicts which turbofan engines will fail within the next 30 cycles, scores each engine's degradation risk, and says **how sure it is** about every prediction. Built on NASA's C-MAPSS run-to-failure data with a hybrid **CNN + BiLSTM + Attention** network, two output heads (fault classifier + continuous risk index) and **Monte Carlo Dropout** uncertainty, benchmarked honestly against simple baselines.

![Risk over engine life](figures/02_risk_over_life.png)

## The problem

Airlines either replace parts on a fixed schedule (wasting usable life) or after something breaks (costly and unsafe). Predictive maintenance uses sensor data to schedule work just before failure. For a maintenance planner a bare prediction isn't enough: they need to know which forecasts to trust and which to send to an engineer for review. That is what the uncertainty layer is for.

## Results (NASA FD001 official test set, 100 unseen engines)

All numbers come from `results/experiments.json` and `results/analysis.json`. Deep models are averaged over 3 seeds, and the decision threshold and seed were chosen on validation engines only.

| Model | Fault PR-AUC | Fault F1 | RUL RMSE (cycles) | RUL RMSE, last window | NASA score ↓ |
|---|---:|---:|---:|---:|---:|
| Threshold on one sensor (s11) | 0.710 | 0.625 | – | – | – |
| Logistic / Ridge | 0.911 | 0.817 | 16.49 | 14.96 | 382 |
| **Random forest** | **0.937** | **0.847** | **14.21** | **12.38** | **247** |
| CNN only | 0.731 | 0.679 | 18.70 | 17.91 | 686 |
| BiLSTM only | 0.905 | 0.805 | 15.62 | 14.71 | 403 |
| CNN + BiLSTM | 0.912 | 0.813 | 15.88 | 15.51 | 516 |
| CNN + BiLSTM + Attention (full) | 0.911 | 0.806 | 14.65 | 14.81 | 442 |

**On point accuracy, the random forest baseline beats the deep model.** That is a common finding on FD001, which has a single operating condition and a single fault mode, and this project reports it rather than hiding it. Within the deep ablation, attention gives the best RUL error.

What the deep model adds is **usable uncertainty** (selected model, 50 MC Dropout passes):

- **Uncertainty tracks error.** Spearman correlation 0.58 between predictive std and absolute error. RUL error rises from 3.6 cycles in the most-confident fifth to 16.7 in the least-confident fifth.
- **Referral works.** Sending the 20% least-certain predictions to a human cuts RUL RMSE on the rest from 13.83 to 11.68 cycles, against 13.87 for random referral.
- **Wrong fault calls are visibly uncertain:** mean std 0.184 when wrong vs 0.006 when right.
- **Every engine was caught early.** On 15 held-out run-to-failure engines, the alarm rule (fault probability ≥ 0.67 for 3 consecutive cycles) caught all 15, with a median of **22 cycles of warning** (range 16–35), no alarms earlier than 60 cycles and none later than 10.
- **Risk categories** (Low / Moderate / High / Critical): 94.6% exact and 100% within one category.
- **Fast enough to deploy.** About 3.7 ms per window for 50 MC passes (compiled, 2 CPU cores), or about 218 windows a second in bulk.

### Honest limitations

- **Intervals are too narrow.** The nominal 95% intervals cover only 70.6% of true values, so treat the std as a ranking signal, not a calibrated interval.
- **Uncertainty misses dataset shift.** On FD003 (a second fault mode the model never saw), RUL RMSE rises to 23.8 cycles but mean uncertainty actually *falls* (0.022 vs 0.038). MC Dropout alone does not flag this kind of shift, so a production system would need a separate out-of-distribution check.
- **Single operating condition.** Trained on FD001 only.

## Tech stack

Python · TensorFlow / Keras (Conv1D, BiLSTM, attention, MC Dropout) · scikit-learn baselines · pandas / NumPy / SciPy · matplotlib · D3.js offline dashboard

## Figures

| Baselines and ablation | Uncertainty vs error | Attention over the window |
|---|---|---|
| ![](figures/03_baselines_ablation.png) | ![](figures/04_uncertainty_vs_error.png) | ![](figures/05_attention_profile.png) |

`dashboard.html` is an interactive, fully offline dashboard of the latest run. Download it and open it in any browser.

## Layout
```
data/                     NASA C-MAPSS FD001 (+ FD003 for the drift test)
src/data.py               loading, targets, unit-level split, train-only scaling, sliding windows
src/model.py              CNN-BiLSTM-Attention (+ ablation variants), training, MC Dropout inference
src/metrics.py            PR-AUC/ROC-AUC/F1/ECE, MAE/RMSE/R2, RUL RMSE, NASA score
src/run_experiments.py    baselines (threshold, logistic/ridge, random forest) + 4-variant ablation x 3 seeds
src/tune.py               validation-only regularisation sweep (documents why the v1 recipe was kept)
src/analyze.py            MC Dropout analysis, risk categories, lead time, drift (FD003), attention,
                          latency, streaming scorer demo, all figures -> then rebuilds dashboard.html
src/make_dashboard.py     builds dashboard.html from results/ (also runs on its own)
src/dashboard_template.html, src/vendor/d3.v7.min.js   dashboard page + charting library (embedded)
results/                  experiments.json, analysis.json, engine_trajectories.csv,
                          attention_profile.csv, streaming_demo.csv
figures/                  report figures
models/                   trained weights + training histories
dashboard.html            interactive dashboard of the latest run (open in any browser, works offline)
```

## Run
```bash
git clone https://github.com/Sharank874/Aircraft-Predictive-Maintenance.git
cd Aircraft-Predictive-Maintenance
pip install -r requirements.txt
python src/data.py              # sanity check: shapes and fault rates
python src/run_experiments.py   # optional: retrain everything (~15-20 min on 2 CPU cores)
python src/analyze.py           # ~5 min; writes results/, figures/ and dashboard.html
```
Then open `dashboard.html` (double-click it). Every run of `analyze.py` rebuilds it from the
new results, so it always shows the latest run. To rebuild it alone: `python src/make_dashboard.py`.

The analysis is seeded, so the same weights give the same uncertainty numbers on every run.

## Key design choices
- Targets: `fault = RUL <= 30`; `risk = 1 - min(RUL,125)/125` (degradation index, not a probability).
- Split by engine (70 train / 15 val / 15 held-out run-to-failure); scaler fit on train engines only.
- Decision threshold and model seed are chosen on validation engines, never on test.
- Test metrics are on the official NASA test set (100 unseen engines).

Data: A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage Propagation Modeling for Aircraft
Engine Run-to-Failure Simulation", PHM 2008 (NASA Prognostics Data Repository).
d3 v7 (ISC licence, see src/vendor/d3.LICENSE) is embedded in the dashboard.
