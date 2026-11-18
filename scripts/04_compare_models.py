"""
Combine the results of all model scripts into one comparison table + charts.
Run from the project folder (after the model scripts):  python scripts/04_compare_models.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
files = sorted((OUT / "results").glob("*.json"))
if not files:
    raise SystemExit("No results found. Run the scripts in scripts/models/ first.")
summary = pd.DataFrame([json.load(open(f)) for f in files])
folds = pd.concat([pd.read_csv(f) for f in (OUT / "results").glob("*_cv_folds.csv")])
summary.to_csv(OUT / "model_comparison.csv", index=False)
folds.to_csv(OUT / "cv_fold_results.csv", index=False)

pd.set_option("display.width", 200)
print(summary[["model", "cv_macro_f1_mean", "train_macro_f1", "test_macro_f1",
               "test_top1_accuracy", "test_log_loss", "test_route_continuity",
               "train_time_s", "inference_ms_per_point"]].round(4).to_string(index=False))

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
print(f"\nCompared {len(summary)} models. Saved: outputs/model_comparison.csv, cv_fold_results.csv, model_comparison.png")
