"""
features.py
Preprocessing, candidate generation, label creation and feature engineering.

For every GPS point we:
  1. find candidate road segments (arcs) within RADIUS metres,
  2. compute features for every (GPS point, candidate arc) pair,
  3. (training only) mark which candidate is the reference arc -> label 1, others 0.
"""
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

from data_loader import load_trip, to_xy

RADIUS_M = 50.0          # candidate search radius in metres
CONTEXT = 4              # neighbouring GPS points used on each side for context features
HEADING_WINDOW = 5       # points on each side used to measure the GPS direction of travel

FEATURE_COLUMNS = [
    "dist", "dist_rank", "dist_minus_min", "n_candidates",
    "heading_diff", "heading_known", "speed", "arc_length", "proj_frac",
    "ctx_dist_mean", "ctx_dist_max", "dt_prev",
]


# ---------------------------------------------------------------- preprocessing
def clean_track(track):
    """Drop missing rows, sort by time, drop repeated timestamps."""
    track = track.dropna().sort_values("t")
    track = track.drop_duplicates(subset="t", keep="first").reset_index(drop=True)
    return track


def prepare_geometry(trip):
    """Project GPS points and road nodes into metres (same origin for both)."""
    track = clean_track(trip["track"])
    lon0, lat0 = track["lon"].mean(), track["lat"].mean()
    px, py = to_xy(track["lon"].values, track["lat"].values, lon0, lat0)
    nx, ny = to_xy(trip["nodes"]["lon"].values, trip["nodes"]["lat"].values, lon0, lat0)
    a = trip["arcs"]["from_node"].values
    b = trip["arcs"]["to_node"].values
    return {
        "track": track, "px": px, "py": py, "t": track["t"].values,
        "ax": nx[a], "ay": ny[a], "bx": nx[b], "by": ny[b],
        "from_node": a, "to_node": b, "origin": (lon0, lat0),
    }


def point_to_segment(px, py, ax, ay, bx, by):
    """Distance (m) from points to segments, plus position along the segment (0..1)."""
    dx, dy = bx - ax, by - ay
    seg_len2 = dx * dx + dy * dy
    u = np.where(seg_len2 > 0, ((px - ax) * dx + (py - ay) * dy) / np.maximum(seg_len2, 1e-12), 0.0)
    u = np.clip(u, 0.0, 1.0)
    cx, cy = ax + u * dx, ay + u * dy
    return np.hypot(px - cx, py - cy), u


# ---------------------------------------------------------------- labels
def align_route(g, route):
    """
    The .route file lists arcs in driving order but does NOT say which GPS point
    belongs to which arc. We assign each GPS point to one route arc so that
    (a) the total point-to-arc distance is as small as possible and
    (b) the route position never goes backwards in time (dynamic programming).
    Returns an arc_id per GPS point.
    """
    T, N = len(g["px"]), len(route)
    ra = route
    D = np.empty(N, dtype=np.float64)
    back = np.empty((T, N), dtype=np.int16 if N < 32000 else np.int32)
    idx = np.arange(N)
    for t in range(T):
        cost, _ = point_to_segment(g["px"][t], g["py"][t], g["ax"][ra], g["ay"][ra], g["bx"][ra], g["by"][ra])
        if t == 0:
            D = cost
            back[0] = idx
            continue
        # best previous position j <= i  (running minimum from the left)
        best_val = np.minimum.accumulate(D)
        is_new_min = np.r_[True, D[1:] < best_val[:-1]]
        best_idx = np.maximum.accumulate(np.where(is_new_min, idx, 0))
        D = cost + best_val
        back[t] = best_idx
    # trace back the best path
    pos = np.empty(T, dtype=np.int64)
    pos[-1] = int(np.argmin(D))
    for t in range(T - 1, 0, -1):
        pos[t - 1] = back[t, pos[t]]
    return route[pos]


# ---------------------------------------------------------------- candidates + features
def build_trip_features(partition, trip_id, data_dir=None, with_labels=True):
    trip = load_trip(partition, trip_id, data_dir)
    g = prepare_geometry(trip)
    T = len(g["px"])

    # --- candidate generation with a spatial index (STRtree) ---
    lines = shapely.linestrings(np.stack([np.c_[g["ax"], g["ay"]], np.c_[g["bx"], g["by"]]], axis=1))
    tree = STRtree(lines)
    points = shapely.points(g["px"], g["py"])
    pt_idx, arc_idx = tree.query(points, predicate="dwithin", distance=RADIUS_M)

    # points with no road within the radius: fall back to the nearest road(s)
    missing = np.setdiff1d(np.arange(T), pt_idx)
    if len(missing):
        m_pt, m_arc = tree.query_nearest(points[missing], return_distance=False)
        pt_idx = np.r_[pt_idx, missing[m_pt]]
        arc_idx = np.r_[arc_idx, m_arc]
    order = np.lexsort((arc_idx, pt_idx))
    pt_idx, arc_idx = pt_idx[order], arc_idx[order]

    ax, ay, bx, by = g["ax"][arc_idx], g["ay"][arc_idx], g["bx"][arc_idx], g["by"][arc_idx]
    dist, u = point_to_segment(g["px"][pt_idx], g["py"][pt_idx], ax, ay, bx, by)

    # --- GPS movement direction and speed (using the previous and next point) ---
    # a wider window (HEADING_WINDOW points each side) gives a much steadier direction at low speed,
    # where GPS jitter of a few metres is as large as the actual movement
    prev_i = np.clip(np.arange(T) - HEADING_WINDOW, 0, T - 1)
    next_i = np.clip(np.arange(T) + HEADING_WINDOW, 0, T - 1)
    mx, my = g["px"][next_i] - g["px"][prev_i], g["py"][next_i] - g["py"][prev_i]
    move = np.hypot(mx, my)
    dt_span = np.maximum(g["t"][next_i] - g["t"][prev_i], 1e-6)
    speed = move / dt_span
    gps_heading = np.degrees(np.arctan2(my, mx))
    heading_known = move > 3.0          # below ~3 m of movement the direction is just noise
    dt_prev = np.r_[0.0, np.diff(g["t"])]

    arc_heading = np.degrees(np.arctan2(by - ay, bx - ax))
    hd = np.abs((gps_heading[pt_idx] - arc_heading + 180) % 360 - 180)   # 0..180 degrees
    hk = heading_known[pt_idx]
    hd = np.where(hk, hd, 90.0)       # unknown direction -> neutral value

    df = pd.DataFrame({
        "trip_id": trip_id, "point": pt_idx, "arc_id": arc_idx,
        "dist": dist, "heading_diff": hd, "heading_known": hk.astype(int),
        "speed": speed[pt_idx], "arc_length": np.hypot(bx - ax, by - ay),
        "proj_frac": u, "dt_prev": np.minimum(dt_prev[pt_idx], 60.0),
        "from_node": g["from_node"][arc_idx], "to_node": g["to_node"][arc_idx],
    })
    grp = df.groupby("point")["dist"]
    df["dist_rank"] = grp.rank(method="first")
    df["dist_minus_min"] = df["dist"] - grp.transform("min")
    df["n_candidates"] = grp.transform("size")

    # --- context: how close are the neighbouring GPS points to the SAME arc? ---
    ctx = []
    for k in [o for o in range(-CONTEXT, CONTEXT + 1) if o != 0]:
        j = np.clip(pt_idx + k, 0, T - 1)
        d_k, _ = point_to_segment(g["px"][j], g["py"][j], ax, ay, bx, by)
        ctx.append(d_k)
    ctx = np.vstack(ctx)
    df["ctx_dist_mean"] = ctx.mean(axis=0)
    df["ctx_dist_max"] = ctx.max(axis=0)

    # --- labels (training / development data only) ---
    if with_labels and trip["route"] is not None:
        true_arc = align_route(g, trip["route"])
        # distance from each GPS point to its reference arc: if it is very large,
        # the reference route does not really cover that part of the track
        d_true, _ = point_to_segment(g["px"], g["py"], g["ax"][true_arc], g["ay"][true_arc],
                                     g["bx"][true_arc], g["by"][true_arc])
        df["true_arc"] = true_arc[pt_idx]
        df["true_dist"] = d_true[pt_idx]
        df["label_reliable"] = (d_true[pt_idx] <= RADIUS_M).astype(int)
        df["label"] = (df["arc_id"] == df["true_arc"]).astype(int)
    return df, g


def build_partition(partition, trip_ids=None, data_dir=None, verbose=True):
    from data_loader import list_trips
    trip_ids = trip_ids or list_trips(partition)
    frames = []
    for n, tid in enumerate(trip_ids, 1):
        df, _ = build_trip_features(partition, tid, data_dir)
        frames.append(df)
        if verbose:
            print(f"[{partition}] {n}/{len(trip_ids)} trip {tid}: {df.point.nunique()} points, {len(df)} candidate rows")
    return pd.concat(frames, ignore_index=True)
