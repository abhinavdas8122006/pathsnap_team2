"""
evaluation.py
Metrics used to compare models.
  - candidate level  : precision, recall, macro F1, log loss  (is this candidate the right road? yes/no)
  - observation level: Top-1 accuracy  (for each GPS point, is the highest-scored candidate the right road?)
  - route level      : route continuity (are consecutive predicted roads the same or connected?)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss


def pick_top1(df, scores):
    """For each GPS point keep the candidate with the highest score."""
    tmp = df[["trip_id", "point", "arc_id", "from_node", "to_node"]].copy()
    if "true_arc" in df:
        tmp["true_arc"] = df["true_arc"].values
    tmp["score"] = scores
    best = tmp.loc[tmp.groupby(["trip_id", "point"])["score"].idxmax()]
    return best.sort_values(["trip_id", "point"])


def route_continuity(top1):
    """Share of consecutive GPS points whose predicted arcs are identical or directly connected."""
    ok, total = 0, 0
    for _, g in top1.groupby("trip_id"):
        arc = g["arc_id"].values
        same = arc[1:] == arc[:-1]
        connected = g["to_node"].values[:-1] == g["from_node"].values[1:]
        ok += np.sum(same | connected)
        total += len(arc) - 1
    return ok / max(total, 1)


def evaluate(df, proba):
    """proba = predicted probability that each candidate row is the correct road."""
    y = df["label"].values
    pred = (proba >= 0.5).astype(int)
    top1 = pick_top1(df, proba)
    return {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro"),
        "log_loss": log_loss(y, np.clip(proba, 1e-7, 1 - 1e-7)),
        "top1_accuracy": float(np.mean(top1["arc_id"].values == top1["true_arc"].values)),
        "route_continuity": route_continuity(top1),
    }
