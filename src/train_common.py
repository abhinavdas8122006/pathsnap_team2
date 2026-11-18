"""
train_common.py
Shared experiment code used by every model script in scripts/models/.
Each model script only defines its model; this file does the rest:
  - loads the feature tables (built by scripts/02_build_features.py)
  - 5-fold GroupKFold cross-validation on the 80 TRAIN trips (grouped by trip)
  - refits on all TRAIN trips and evaluates on the 10 STUDENT_TEST trips
  - saves the model, its metrics (JSON), fold results and confusion matrix
"""
import json, time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from features import FEATURE_COLUMNS
from evaluation import evaluate

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
RESULTS = OUT / "results"
MODELS = ROOT / "models"
SEED = 42
POINT_SAMPLE = 0.30      # 30% of training GPS points (with all their candidates) to keep training fast


def load_data():
    train = pd.read_pickle(OUT / "features_train.pkl")
    test = pd.read_pickle(OUT / "features_student_test.pkl")
    train = train[train.label_reliable == 1]          # drop points the reference route does not cover
    test = test[test.label_reliable == 1]
    points = train[["trip_id", "point"]].drop_duplicates()
    keep = points.sample(frac=POINT_SAMPLE, random_state=SEED)
    train = train.merge(keep, on=["trip_id", "point"]).reset_index(drop=True)
    return train, test.reset_index(drop=True)


def run_experiment(name, family, config, model):
    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    train, test = load_data()
    print(f"[{name}] training rows: {len(train):,} from {train.trip_id.nunique()} trips | test rows: {len(test):,}")
    X, y, groups = train[FEATURE_COLUMNS].values, train["label"].values, train["trip_id"].values
    X_test = test[FEATURE_COLUMNS].values

    # ---- 5-fold group-aware cross-validation
    fold_rows = []
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=5).split(X, y, groups), 1):
        model.fit(X[tr], y[tr])
        m = evaluate(train.iloc[va], model.predict_proba(X[va])[:, 1])
        m.update(model=name, fold=fold)
        fold_rows.append(m)
        print(f"  fold {fold}: macro F1={m['macro_f1']:.4f}  top1={m['top1_accuracy']:.4f}")
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(RESULTS / f"{name}_cv_folds.csv", index=False)

    # ---- refit on all training data, evaluate on student_test
    t0 = time.time()
    model.fit(X, y)
    train_time = time.time() - t0
    train_m = evaluate(train, model.predict_proba(X)[:, 1])

    t0 = time.time()
    p_test = model.predict_proba(X_test)[:, 1]
    n_points = test.groupby(["trip_id", "point"]).ngroups
    infer_ms = (time.time() - t0) / n_points * 1000
    test_m = evaluate(test, p_test)

    result = {
        "model": name, "family": family, "config": config,
        "cv_macro_f1_mean": folds.macro_f1.mean(), "cv_macro_f1_std": folds.macro_f1.std(),
        "cv_top1_mean": folds.top1_accuracy.mean(),
        "train_macro_f1": train_m["macro_f1"],
        **{f"test_{k}": v for k, v in test_m.items()},
        "train_time_s": train_time, "inference_ms_per_point": infer_ms,
        "seed": SEED, "point_sample": POINT_SAMPLE,
    }
    with open(RESULTS / f"{name}.json", "w") as f:
        json.dump({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in result.items()}, f, indent=2)
    joblib.dump(model, MODELS / f"{name}.joblib")

    cm = confusion_matrix(test["label"], (p_test >= 0.5).astype(int))
    ConfusionMatrixDisplay(cm, display_labels=["wrong road", "correct road"]).plot(cmap="Blues")
    plt.title(f"{name} - student_test confusion matrix")
    plt.savefig(OUT / f"cm_{name}.png", dpi=110, bbox_inches="tight")
    plt.close()

    print(f"\n[{name}] CV macro F1 = {result['cv_macro_f1_mean']:.4f} ± {result['cv_macro_f1_std']:.4f}")
    print(f"[{name}] student_test: macro F1 = {test_m['macro_f1']:.4f}, top-1 = {test_m['top1_accuracy']:.4f}, "
          f"log loss = {test_m['log_loss']:.4f}, route continuity = {test_m['route_continuity']:.4f}")
    print(f"[{name}] train time = {train_time:.1f}s, inference = {infer_ms:.4f} ms/point")
    print(f"Saved: models/{name}.joblib, outputs/results/{name}.json, outputs/cm_{name}.png")
    return model, result
