"""
Step 5: Bayesian hyperparameter optimisation of the chosen model (HistGradientBoosting) with Optuna.

- Sampler: TPE (Tree-structured Parzen Estimator) = a Bayesian optimisation method.
  It builds a probability model of "which settings give good scores" from earlier trials
  and uses it to choose the next settings to try (instead of random or grid search).
- Objective: mean 5-fold GroupKFold macro F1 on the 80 TRAIN trips (same folds as untuned model).
- student_test is used ONLY after tuning, to compare tuned vs untuned. Evaluator data is never used.

Run from the project folder:   python scripts/05_bayesian_optimisation.py
"""
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import joblib
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier

from features import FEATURE_COLUMNS
from evaluation import evaluate
from train_common import load_data, SEED, OUT, RESULTS, MODELS

N_TRIALS = 20
OPT = OUT / "optimisation"
OPT.mkdir(parents=True, exist_ok=True)

train, test = load_data()
X, y, groups = train[FEATURE_COLUMNS].values, train["label"].values, train["trip_id"].values
X_test = test[FEATURE_COLUMNS].values
folds = list(GroupKFold(n_splits=5).split(X, y, groups))


def cv_score(params):
    scores, top1 = [], []
    for tr, va in folds:
        model = HistGradientBoostingClassifier(**params, random_state=SEED)
        model.fit(X[tr], y[tr])
        m = evaluate(train.iloc[va], model.predict_proba(X[va])[:, 1])
        scores.append(m["macro_f1"])
        top1.append(m["top1_accuracy"])
    return float(np.mean(scores)), float(np.std(scores)), float(np.mean(top1))


def objective(trial):
    # ---- the search space
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        "max_iter": trial.suggest_int("max_iter", 100, 500, step=50),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 300),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
    }
    start = time.time()
    mean_f1, std_f1, mean_top1 = cv_score(params)
    trial.set_user_attr("cv_macro_f1_std", std_f1)
    trial.set_user_attr("cv_top1_mean", mean_top1)
    trial.set_user_attr("seconds", time.time() - start)
    print(f"trial {trial.number:2d}: CV macro F1={mean_f1:.4f}  top1={mean_top1:.4f}  {params}")
    return mean_f1


optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
t0 = time.time()
study.optimize(objective, n_trials=N_TRIALS)
search_seconds = time.time() - t0

# ---- save the full search record (required by the rules)
trials = study.trials_dataframe()
trials.to_csv(OPT / "optuna_trials.csv", index=False)
best = study.best_params
with open(OPT / "best_params.json", "w") as f:
    json.dump({"best_params": best, "best_cv_macro_f1": study.best_value,
               "n_trials": N_TRIALS, "search_seconds": search_seconds, "sampler": "TPE (seed 42)"}, f, indent=2)
print(f"\nBest CV macro F1 = {study.best_value:.4f} with {best}  (search took {search_seconds/60:.1f} min)")

# ---- refit the tuned model on all training data and test on student_test
tuned = HistGradientBoostingClassifier(**best, random_state=SEED)
t0 = time.time(); tuned.fit(X, y); train_time = time.time() - t0
t0 = time.time(); p = tuned.predict_proba(X_test)[:, 1]
infer_ms = (time.time() - t0) / test.groupby(["trip_id", "point"]).ngroups * 1000
test_m = evaluate(test, p)
joblib.dump(tuned, MODELS / "HistGradientBoosting_tuned.joblib")

best_trial = study.best_trial
tuned_result = {
    "model": "HistGradientBoosting_tuned", "config": json.dumps(best),
    "cv_macro_f1_mean": study.best_value, "cv_macro_f1_std": best_trial.user_attrs["cv_macro_f1_std"],
    "cv_top1_mean": best_trial.user_attrs["cv_top1_mean"],
    **{f"test_{k}": float(v) for k, v in test_m.items()},
    "train_time_s": train_time, "inference_ms_per_point": infer_ms,
}
with open(RESULTS.parent / "optimisation" / "HistGradientBoosting_tuned.json", "w") as f:
    json.dump(tuned_result, f, indent=2)

# ---- untuned vs tuned table
untuned = json.load(open(RESULTS / "HistGradientBoosting.json"))
cols = ["cv_macro_f1_mean", "test_top1_accuracy", "test_macro_f1", "test_log_loss",
        "test_route_continuity", "train_time_s", "inference_ms_per_point"]
comp = pd.DataFrame([{**{"model": "untuned"}, **{c: untuned[c] for c in cols}},
                     {**{"model": "tuned"}, **{c: tuned_result[c] for c in cols}}])
comp.to_csv(OPT / "tuned_vs_untuned.csv", index=False)
pd.set_option("display.width", 200)
print("\n", comp.round(4).to_string(index=False))

# ---- plots: optimisation history + tuned vs untuned
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
vals = trials["value"].values
axes[0].plot(vals, "o", alpha=0.6, label="trial score")
axes[0].plot(np.maximum.accumulate(vals), "r-", label="best so far")
axes[0].axhline(untuned["cv_macro_f1_mean"], color="grey", ls="--", label="untuned model")
axes[0].set_xlabel("trial"); axes[0].set_ylabel("CV macro F1"); axes[0].legend()
axes[0].set_title("Bayesian optimisation history (Optuna TPE)")
metrics = ["cv_macro_f1_mean", "test_top1_accuracy", "test_macro_f1"]
xx = np.arange(len(metrics))
axes[1].bar(xx - 0.2, [untuned[m] for m in metrics], 0.4, label="untuned")
axes[1].bar(xx + 0.2, [tuned_result[m] for m in metrics], 0.4, label="tuned")
axes[1].set_xticks(xx, ["CV macro F1", "test Top-1", "test macro F1"])
axes[1].set_ylim(min(untuned[m] for m in metrics) - 0.03, 1.0)
axes[1].legend(); axes[1].set_title("Untuned vs tuned HistGradientBoosting")
plt.tight_layout()
plt.savefig(OPT / "optimisation_results.png", dpi=120)
print("\nSaved: outputs/optimisation/ (optuna_trials.csv, best_params.json, tuned_vs_untuned.csv, "
      "optimisation_results.png) and models/HistGradientBoosting_tuned.joblib")
