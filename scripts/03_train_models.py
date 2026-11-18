"""
Step 4: train and compare 5 models.
  - 5-fold GroupKFold cross-validation on the 80 TRAIN trips (grouped by trip)
  - refit on all TRAIN trips, evaluate on the 10 STUDENT_TEST trips
Run from the project folder:   python scripts/03_train_models.py
"""
import sys, time, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from features import FEATURE_COLUMNS
from evaluation import evaluate

SEED = 42
POINT_SAMPLE = 0.30      # use 30% of training GPS points (all their candidates) to keep training fast
OUT, MODELS = ROOT / "outputs", ROOT / "models"
MODELS.mkdir(exist_ok=True)

# ------------------------------------------------------------------ data
train = pd.read_pickle(OUT / "features_train.pkl")
test = pd.read_pickle(OUT / "features_student_test.pkl")
train = train[train.label_reliable == 1]          # drop points the reference route does not cover
test = test[test.label_reliable == 1]

rng = np.random.default_rng(SEED)
points = train[["trip_id", "point"]].drop_duplicates()
keep = points.sample(frac=POINT_SAMPLE, random_state=SEED)
train = train.merge(keep, on=["trip_id", "point"]).reset_index(drop=True)
print(f"Training rows: {len(train):,} from {train.trip_id.nunique()} trips | test rows: {len(test):,}")

X, y, groups = train[FEATURE_COLUMNS].values, train["label"].values, train["trip_id"].values
X_test = test[FEATURE_COLUMNS].values

# ------------------------------------------------------------------ the 5 models
MODELS_TO_TRY = {
    "LogisticRegression": ("Linear", make_pipeline(StandardScaler(),
                            LogisticRegression(C=1.0, max_iter=1000))),
    "GaussianNB": ("Probabilistic", make_pipeline(StandardScaler(), GaussianNB())),
    "RandomForest": ("Ensemble (bagging)", RandomForestClassifier(
                            n_estimators=60, max_depth=12, min_samples_leaf=20,
                            n_jobs=-1, random_state=SEED)),
    "HistGradientBoosting": ("Boosting", HistGradientBoostingClassifier(
                            max_iter=200, learning_rate=0.1, max_leaf_nodes=31, random_state=SEED)),
    "MLP": ("Neural network", make_pipeline(StandardScaler(), MLPClassifier(
                            hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=512,
                            max_iter=40, early_stopping=True, random_state=SEED))),
}
CONFIGS = {
    "LogisticRegression": "C=1.0, L2, scaled",
    "GaussianNB": "default var_smoothing, scaled",
    "RandomForest": "60 trees, max_depth=12, min_leaf=20",
    "HistGradientBoosting": "200 iters, lr=0.1, 31 leaves",
    "MLP": "64-32 ReLU, Adam, early stopping",
}

# ------------------------------------------------------------------ cross-validation + test
gkf = GroupKFold(n_splits=5)
fold_rows, summary = [], []
for name, (family, model) in MODELS_TO_TRY.items():
    print(f"\n=== {name} ===")
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups), 1):
        model.fit(X[tr], y[tr])
        m = evaluate(train.iloc[va], model.predict_proba(X[va])[:, 1])
        m.update(model=name, fold=fold)
        fold_rows.append(m)
        print(f"  fold {fold}: macro F1={m['macro_f1']:.4f}  top1={m['top1_accuracy']:.4f}")

    t0 = time.time()
    model.fit(X, y)
    train_time = time.time() - t0
    train_m = evaluate(train, model.predict_proba(X)[:, 1])

    t0 = time.time()
    p_test = model.predict_proba(X_test)[:, 1]
    infer_ms = (time.time() - t0) / test.groupby(["trip_id", "point"]).ngroups * 1000
    test_m = evaluate(test, p_test)

    cv = pd.DataFrame([r for r in fold_rows if r["model"] == name])
    summary.append({
        "model": name, "family": family, "config": CONFIGS[name],
        "cv_macro_f1_mean": cv.macro_f1.mean(), "cv_macro_f1_std": cv.macro_f1.std(),
        "cv_top1_mean": cv.top1_accuracy.mean(),
        "train_macro_f1": train_m["macro_f1"],
        **{f"test_{k}": v for k, v in test_m.items()},
        "train_time_s": train_time, "inference_ms_per_point": infer_ms,
    })
    joblib.dump(model, MODELS / f"{name}.joblib")

    cm = confusion_matrix(test["label"], (p_test >= 0.5).astype(int))
    ConfusionMatrixDisplay(cm, display_labels=["wrong road", "correct road"]).plot(cmap="Blues")
    plt.title(f"{name} - student_test confusion matrix")
    plt.savefig(OUT / f"cm_{name}.png", dpi=110, bbox_inches="tight")
    plt.close()

    if name == "MLP":   # only the MLP produces an epoch-wise training history
        mlp = model[-1]
        fig, ax1 = plt.subplots()
        ax1.plot(mlp.loss_curve_, label="training loss")
        ax1.set_xlabel("epoch"); ax1.set_ylabel("training log loss")
        ax2 = ax1.twinx()
        ax2.plot(mlp.validation_scores_, color="orange", label="validation accuracy")
        ax2.set_ylabel("validation accuracy")
        fig.legend(loc="upper center"); plt.title("MLP training history")
        plt.savefig(OUT / "mlp_training_curve.png", dpi=110, bbox_inches="tight")
        plt.close()

folds = pd.DataFrame(fold_rows)
folds.to_csv(OUT / "cv_fold_results.csv", index=False)
summary = pd.DataFrame(summary)
summary.to_csv(OUT / "model_comparison.csv", index=False)
pd.set_option("display.width", 200)
print("\n", summary[["model", "cv_macro_f1_mean", "train_macro_f1", "test_macro_f1",
                     "test_top1_accuracy", "test_log_loss", "test_route_continuity",
                     "train_time_s", "inference_ms_per_point"]].round(4).to_string(index=False))

# ------------------------------------------------------------------ comparison plots
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
axes[0].bar(summary.model, summary.cv_macro_f1_mean, yerr=summary.cv_macro_f1_std, capsize=4)
axes[0].set_title("Cross-validation macro F1 (train, 5 folds)")
x = np.arange(len(summary))
axes[1].bar(x - 0.2, summary.test_top1_accuracy, 0.4, label="Top-1 accuracy")
axes[1].bar(x + 0.2, summary.test_macro_f1, 0.4, label="Macro F1")
axes[1].set_xticks(x, summary.model); axes[1].legend(); axes[1].set_title("Student-test performance")
axes[2].bar(summary.model, summary.test_log_loss, color="tab:red")
axes[2].set_title("Student-test log loss (lower is better)")
for ax in axes:
    ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig(OUT / "model_comparison.png", dpi=120)
print("\nSaved: outputs/model_comparison.csv, cv_fold_results.csv, model_comparison.png, cm_*.png, models/*.joblib")
