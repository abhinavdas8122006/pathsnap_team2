"""
Step 6: HMM + Viterbi route reconstruction, and classifier-only vs classifier+HMM comparison.

1. Decoder settings (beta, emission_weight) are tuned on 20 TRAIN trips using OUT-OF-FOLD
   classifier probabilities (the classifier never saw the trip it is scoring) -> no leakage.
2. The chosen settings are evaluated on the 10 STUDENT_TEST trips.
3. Saves comparison tables, a per-trip chart and an interactive Folium map.

Run from the project folder:   python scripts/06_hmm_viterbi.py
"""
import sys, json, time, itertools
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier

from data_loader import list_trips, load_trip
from features import build_trip_features, FEATURE_COLUMNS
from hmm import viterbi_decode, route_from_points, DEFAULTS
from train_common import load_data, SEED, OUT, MODELS

HMM_OUT = OUT / "hmm"
HMM_OUT.mkdir(parents=True, exist_ok=True)
GRID = {"beta": [3, 5, 8, 12], "emission_weight": [0.5, 1.0]}
N_TUNE_TRIPS = 20

# ------------------------------------------------------------------ which classifier
tuned_path = MODELS / "HistGradientBoosting_tuned.joblib"
model_path = tuned_path if tuned_path.exists() else MODELS / "HistGradientBoosting.joblib"
model = joblib.load(model_path)
print(f"Classifier: {model_path.name}")
best_params_file = OUT / "optimisation" / "best_params.json"
clf_params = (json.load(open(best_params_file))["best_params"] if tuned_path.exists() and best_params_file.exists()
              else dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=31))


def point_metrics(df, chosen, g):
    """Top-1 accuracy and route continuity for one trip's per-point predictions."""
    truth = df.groupby("point")["true_arc"].first()
    rel = df.groupby("point")["label_reliable"].max() == 1
    pts = truth.index[rel].values
    correct = int(np.sum(chosen[pts] == truth[rel].values))
    seq = chosen[chosen >= 0]
    same = seq[1:] == seq[:-1]
    connected = g["to_node"][seq[:-1]] == g["from_node"][seq[1:]]
    return correct, len(pts), int(np.sum(same | connected)), len(seq) - 1


def classifier_only(df, proba):
    tmp = df[["point", "arc_id"]].copy()
    tmp["p"] = proba
    best = tmp.loc[tmp.groupby("point")["p"].idxmax()]
    chosen = np.full(int(df.point.max()) + 1, -1, dtype=np.int64)
    chosen[best.point.values] = best.arc_id.values
    return chosen


def route_f1(pred_route, true_route):
    """Sequence-level: overlap between the set of predicted route arcs and reference route arcs."""
    p, t = set(pred_route.tolist()), set(true_route.tolist())
    tp = len(p & t)
    prec = tp / max(len(p), 1); rec = tp / max(len(t), 1)
    return 2 * prec * rec / max(prec + rec, 1e-9)


# ------------------------------------------------------------------ 1. tune decoder on train trips (out-of-fold)
train, _ = load_data()
rng = np.random.default_rng(SEED)
tune_trips = sorted(rng.choice(list_trips("train"), N_TUNE_TRIPS, replace=False))
tune_data = []
for fold in range(4):
    held = tune_trips[fold::4]
    fit_rows = train[~train.trip_id.isin(held)]
    clf = HistGradientBoostingClassifier(**clf_params, random_state=SEED)
    clf.fit(fit_rows[FEATURE_COLUMNS].values, fit_rows["label"].values)
    for tid in held:
        df, g = build_trip_features("train", tid)
        tune_data.append((df, clf.predict_proba(df[FEATURE_COLUMNS].values)[:, 1], g))
print(f"Tuning decoder on {N_TUNE_TRIPS} train trips (out-of-fold probabilities)")

grid_rows = []
for beta, ew in itertools.product(GRID["beta"], GRID["emission_weight"]):
    c = n = 0
    for df, pr, g in tune_data:
        ch = viterbi_decode(df, pr, g, {"beta": beta, "emission_weight": ew})
        cc, nn, _, _ = point_metrics(df, ch, g)
        c += cc; n += nn
    grid_rows.append({"beta": beta, "emission_weight": ew, "train_oof_top1": c / n})
    print(f"  beta={beta:>2}  emission_weight={ew}:  top-1 = {c / n:.4f}")
grid = pd.DataFrame(grid_rows)
grid.to_csv(HMM_OUT / "decoder_grid_search.csv", index=False)
best = grid.loc[grid.train_oof_top1.idxmax()]
hmm_params = {**DEFAULTS, "beta": float(best.beta), "emission_weight": float(best.emission_weight)}
json.dump(hmm_params, open(MODELS / "hmm_config.json", "w"), indent=2)
print(f"Chosen decoder settings: beta={best.beta}, emission_weight={best.emission_weight}")

# ------------------------------------------------------------------ 2. evaluate on student_test
rows = []
maps_data = {}
for tid in list_trips("student_test"):
    df, g = build_trip_features("student_test", tid)
    proba = model.predict_proba(df[FEATURE_COLUMNS].values)[:, 1]
    true_route = load_trip("student_test", tid)["route"]

    ch_cls = classifier_only(df, proba)
    t0 = time.time()
    ch_hmm = viterbi_decode(df, proba, g, hmm_params)
    decode_s = time.time() - t0
    for method, ch in [("classifier_only", ch_cls), ("classifier_plus_hmm", ch_hmm)]:
        c, n, ok, tot = point_metrics(df, ch, g)
        rows.append({"trip_id": tid, "method": method, "points": n, "top1_accuracy": c / n,
                     "route_continuity": ok / max(tot, 1),
                     "route_f1": route_f1(route_from_points(ch), true_route),
                     "decode_seconds": decode_s if method == "classifier_plus_hmm" else 0.0})
    maps_data[tid] = (g, ch_cls, ch_hmm, true_route)

per_trip = pd.DataFrame(rows)
per_trip.to_csv(HMM_OUT / "hmm_per_trip.csv", index=False)
summary = per_trip.groupby("method").apply(lambda d: pd.Series({
    "top1_accuracy": np.average(d.top1_accuracy, weights=d.points),
    "route_continuity": d.route_continuity.mean(),
    "route_f1": d.route_f1.mean(),
}), include_groups=False).reset_index()
summary.to_csv(HMM_OUT / "classifier_vs_hmm.csv", index=False)
pd.set_option("display.width", 200)
print("\nStudent-test, per trip:\n", per_trip.pivot(index="trip_id", columns="method", values="top1_accuracy").round(4))
print("\nStudent-test, overall:\n", summary.round(4).to_string(index=False))

# ------------------------------------------------------------------ 3. chart
piv = per_trip.pivot(index="trip_id", columns="method", values="top1_accuracy")
ax = piv[["classifier_only", "classifier_plus_hmm"]].plot.bar(figsize=(12, 4.5), width=0.8)
ax.set_ylabel("Top-1 accuracy"); ax.set_ylim(0.6, 1.0)
ax.set_title("Student-test Top-1 accuracy per trip: classifier only vs classifier + HMM/Viterbi")
plt.tight_layout(); plt.savefig(HMM_OUT / "classifier_vs_hmm.png", dpi=120)

# ------------------------------------------------------------------ 4. Folium map for one trip
import folium
tid = list_trips("student_test")[0]
g, ch_cls, ch_hmm, true_route = maps_data[tid]
trip = load_trip("student_test", tid)
nodes, track = trip["nodes"], g["track"]
arcs = trip["arcs"]
m = folium.Map(location=[track.lat.mean(), track.lon.mean()], zoom_start=15, tiles="OpenStreetMap")

def add_route(arc_ids, name, colour, weight, dash=None):
    layer = folium.FeatureGroup(name=name)
    for a in arc_ids:
        f, t = arcs.from_node[a], arcs.to_node[a]
        folium.PolyLine([(nodes.lat[f], nodes.lon[f]), (nodes.lat[t], nodes.lon[t])],
                        color=colour, weight=weight, opacity=0.8, dash_array=dash).add_to(layer)
    layer.add_to(m)

add_route(true_route, "Reference route (green)", "green", 8)
add_route(route_from_points(ch_cls), "Classifier only (orange)", "orange", 4, "6")
add_route(route_from_points(ch_hmm), "Classifier + HMM (red)", "red", 4)
gps = folium.FeatureGroup(name="Raw GPS points (blue)")
for _, p in track.iloc[::3].iterrows():
    folium.CircleMarker((p.lat, p.lon), radius=2, color="blue", fill=True).add_to(gps)
gps.add_to(m)
folium.LayerControl(collapsed=False).add_to(m)
m.save(str(HMM_OUT / f"map_hmm_student_test_{tid}.html"))
print(f"\nSaved: outputs/hmm/ (decoder_grid_search.csv, hmm_per_trip.csv, classifier_vs_hmm.csv, "
      f"classifier_vs_hmm.png, map_hmm_student_test_{tid}.html) and models/hmm_config.json")
