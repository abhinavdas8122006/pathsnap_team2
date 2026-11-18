"""
PathSnap API  (runs on LAPTOP 1)

Endpoints
  GET  /health          -> is the server up, which model is loaded
  GET  /trips           -> the trip IDs available for evaluation (no ground truth)
  POST /predict         -> {"trip_id": "..."} : runs the full pipeline and returns the predicted route
                           + accuracy computed SERVER-SIDE against the reference route

Ground truth never leaves this laptop: the prediction step runs with labels switched off,
and only summary accuracy numbers (not the reference route) are sent back to the UI.

Start (inside pathsnap, venv active):
  uvicorn api.main:app --host 0.0.0.0 --port 8000
Choose which trips to serve (default: data/evaluator_test, falls back to data/student_test):
  PATHSNAP_PARTITION=student_test uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
import os, sys, json, time, logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from data_loader import DATA_DIR, load_trip, to_xy
from features import build_trip_features, align_route, point_to_segment, FEATURE_COLUMNS
from hmm import viterbi_decode, route_from_points, DEFAULTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap")

# ------------------------------------------------------------------ load artifacts once at start-up
MODEL_DIR = ROOT / "models"
model_file = MODEL_DIR / "HistGradientBoosting_tuned.joblib"
if not model_file.exists():
    model_file = MODEL_DIR / "HistGradientBoosting.joblib"
MODEL = joblib.load(model_file)
hmm_file = MODEL_DIR / "hmm_config.json"
HMM_PARAMS = json.load(open(hmm_file)) if hmm_file.exists() else DEFAULTS

PARTITION = os.environ.get("PATHSNAP_PARTITION", "evaluator_test")
if not (DATA_DIR / PARTITION).exists():
    log.warning("data/%s not found - falling back to data/student_test", PARTITION)
    PARTITION = "student_test"
log.info("Loaded model %s | HMM %s | serving trips from data/%s", model_file.name, HMM_PARAMS, PARTITION)

app = FastAPI(title="PathSnap map-matching API")


class PredictRequest(BaseModel):
    trip_id: str


def available_trips():
    folder = DATA_DIR / PARTITION
    return sorted(p.name for p in folder.iterdir() if p.is_dir() and (p / f"{p.name}.track").exists())


@app.get("/health")
def health():
    return {"status": "ok", "model": model_file.name, "partition": PARTITION, "hmm": HMM_PARAMS}


@app.get("/trips")
def trips():
    ids = available_trips()
    log.info("GET /trips -> %d trips", len(ids))
    return {"partition": PARTITION, "trips": ids}


def evaluate_server_side(trip_id, g, chosen_hmm, chosen_cls):
    """Compare predictions with the reference route. Runs only AFTER prediction is finished."""
    route = load_trip(PARTITION, trip_id)["route"]
    if route is None:
        return None
    true_arc = align_route(g, route)
    d_true, _ = point_to_segment(g["px"], g["py"], g["ax"][true_arc], g["ay"][true_arc],
                                 g["bx"][true_arc], g["by"][true_arc])
    reliable = d_true <= 50.0
    pred_set, true_set = set(route_from_points(chosen_hmm).tolist()), set(route.tolist())
    tp = len(pred_set & true_set)
    prec, rec = tp / max(len(pred_set), 1), tp / max(len(true_set), 1)
    return {
        "top1_accuracy_hmm": float(np.mean(chosen_hmm[reliable] == true_arc[reliable])),
        "top1_accuracy_classifier_only": float(np.mean(chosen_cls[reliable] == true_arc[reliable])),
        "route_precision": prec, "route_recall": rec,
        "route_f1": 2 * prec * rec / max(prec + rec, 1e-9),
        "points_evaluated": int(reliable.sum()),
    }


@app.post("/predict")
def predict(req: PredictRequest):
    trip_id = req.trip_id.strip()
    if trip_id not in available_trips():
        raise HTTPException(status_code=404, detail=f"Unknown trip {trip_id}")
    log.info("POST /predict trip=%s : started", trip_id)
    t0 = time.time()

    # 1) preprocessing + candidates + features   (labels OFF: ground truth not used for prediction)
    df, g = build_trip_features(PARTITION, trip_id, with_labels=False)
    # 2) classifier probabilities
    proba = MODEL.predict_proba(df[FEATURE_COLUMNS].values)[:, 1]
    # 3) classifier-only choice (for comparison) and HMM/Viterbi route
    tmp = df[["point", "arc_id"]].assign(p=proba)
    best = tmp.loc[tmp.groupby("point")["p"].idxmax()]
    chosen_cls = np.full(len(g["px"]), -1, dtype=np.int64)
    chosen_cls[best.point.values] = best.arc_id.values
    chosen_hmm = viterbi_decode(df, proba, g, HMM_PARAMS)
    predict_seconds = time.time() - t0

    # 4) server-side accuracy (reference route stays on this laptop)
    metrics = evaluate_server_side(trip_id, g, chosen_hmm, chosen_cls)

    # 5) geometry for the UI map: GPS points + predicted route segments (lat/lon)
    trip = load_trip(PARTITION, trip_id)
    nodes, arcs = trip["nodes"], trip["arcs"]
    route = route_from_points(chosen_hmm)
    segments = [[[float(nodes.lat[arcs.from_node[a]]), float(nodes.lon[arcs.from_node[a]])],
                 [float(nodes.lat[arcs.to_node[a]]), float(nodes.lon[arcs.to_node[a]])]] for a in route]
    track = g["track"]
    step = max(1, len(track) // 1500)
    gps = track.iloc[::step][["lat", "lon"]].values.tolist()

    seq = chosen_hmm[chosen_hmm >= 0]
    continuity = float(np.mean((seq[1:] == seq[:-1]) | (g["to_node"][seq[:-1]] == g["from_node"][seq[1:]])))
    log.info("POST /predict trip=%s : %d GPS points, %d route arcs, %.2fs, metrics=%s",
             trip_id, len(track), len(route), predict_seconds,
             {k: round(v, 4) for k, v in (metrics or {}).items()})
    return {
        "trip_id": trip_id, "n_gps_points": len(track), "n_route_arcs": int(len(route)),
        "processing_seconds": round(predict_seconds, 3), "route_continuity": continuity,
        "metrics": metrics, "gps_points": gps, "predicted_route_segments": segments,
        "model": model_file.name,
    }
