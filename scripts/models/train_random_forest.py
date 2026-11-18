"""Model 3 - Random Forest (bagging ensemble family).
Run from the project folder:  python scripts/models/train_random_forest.py"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.ensemble import RandomForestClassifier
from train_common import run_experiment, SEED

# Many decision trees on bootstrap samples, averaged. max_depth / min_samples_leaf limit overfitting.
model = RandomForestClassifier(n_estimators=60, max_depth=12, min_samples_leaf=20,
                               n_jobs=-1, random_state=SEED)
run_experiment("RandomForest", "Ensemble (bagging)", "60 trees, max_depth=12, min_leaf=20", model)
