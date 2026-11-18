"""Model 1 - Logistic Regression (linear family).
Run from the project folder:  python scripts/models/train_logistic_regression.py"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from train_common import run_experiment

# Scaling matters: logistic regression is sensitive to feature ranges (metres vs degrees vs seconds).
model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
run_experiment("LogisticRegression", "Linear", "C=1.0, L2, scaled", model)
