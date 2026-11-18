"""Model 4 - Histogram Gradient Boosting (boosting family).
Run from the project folder:  python scripts/models/train_hist_gradient_boosting.py"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.ensemble import HistGradientBoostingClassifier
from train_common import run_experiment, SEED

# Trees built one after another, each correcting the previous trees' errors. Optimises log loss.
model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                       random_state=SEED)
run_experiment("HistGradientBoosting", "Boosting", "200 iters, lr=0.1, 31 leaves", model)
