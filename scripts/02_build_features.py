"""
Step 3: build candidate + feature tables for TRAIN and STUDENT_TEST.
Run from the project folder:   python scripts/02_build_features.py
"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from features import build_partition, RADIUS_M

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

stats = []
for partition in ["train", "student_test"]:
    start = time.time()
    df = build_partition(partition)
    df.to_pickle(OUT / f"features_{partition}.pkl")
    per_point = df.groupby(["trip_id", "point"])["label"].max()
    reliable = df.groupby(["trip_id", "point"])["label_reliable"].max()
    stats.append({
        "partition": partition,
        "gps_points": len(per_point),
        "candidate_rows": len(df),
        "avg_candidates_per_point": round(len(df) / len(per_point), 2),
        "positive_share": round(df["label"].mean(), 4),
        "candidate_miss_rate_all": round(1 - per_point.mean(), 4),
        "points_with_unreliable_label": int((reliable == 0).sum()),
        "candidate_miss_rate_reliable": round(1 - per_point[reliable == 1].mean(), 4),
        "radius_m": RADIUS_M,
        "build_seconds": round(time.time() - start, 1),
    })

stats = pd.DataFrame(stats)
stats.to_csv(OUT / "candidate_stats.csv", index=False)
print(stats.to_string(index=False))
