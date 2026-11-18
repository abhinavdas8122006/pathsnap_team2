"""Model 2 - Gaussian Naive Bayes (probabilistic family).
Run from the project folder:  python scripts/models/train_gaussian_nb.py"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from train_common import run_experiment

# Assumes every feature is independent and normally distributed within each class.
model = make_pipeline(StandardScaler(), GaussianNB())
run_experiment("GaussianNB", "Probabilistic", "default var_smoothing, scaled", model)
