"""Model 5 - Multi-layer Perceptron (neural network family).
Run from the project folder:  python scripts/models/train_mlp.py"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from train_common import run_experiment, SEED, OUT

model = make_pipeline(StandardScaler(), MLPClassifier(
    hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=512,
    max_iter=40, early_stopping=True, random_state=SEED))
model, _ = run_experiment("MLP", "Neural network", "64-32 ReLU, Adam, early stopping", model)

# Only the MLP trains in epochs, so only it gets a training-history plot
mlp = model[-1]
fig, ax1 = plt.subplots()
ax1.plot(mlp.loss_curve_, label="training loss")
ax1.set_xlabel("epoch"); ax1.set_ylabel("training log loss")
ax2 = ax1.twinx()
ax2.plot(mlp.validation_scores_, color="orange", label="validation accuracy")
ax2.set_ylabel("validation accuracy")
fig.legend(loc="upper center"); plt.title("MLP training history")
plt.savefig(OUT / "mlp_training_curve.png", dpi=110, bbox_inches="tight")
print("Saved: outputs/mlp_training_curve.png")
