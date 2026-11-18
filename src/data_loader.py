"""
data_loader.py
Reads the raw PathSnap files for one trip and converts coordinates to metres.
"""
from pathlib import Path
import numpy as np
import pandas as pd

# The "data" folder sits next to the "src" folder
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EARTH_RADIUS_M = 6_371_000.0


def list_trips(partition):
    """Return all trip IDs (folder names) in a partition, e.g. 'train' or 'student_test'."""
    folder = DATA_DIR / partition
    return sorted(p.name for p in folder.iterdir() if p.is_dir())


def load_trip(partition, trip_id, data_dir=None):
    """Load the 4 files of one trip into pandas tables."""
    base = Path(data_dir) if data_dir else DATA_DIR
    folder = base / partition / trip_id

    track = pd.read_csv(folder / f"{trip_id}.track", sep=r"\s+", header=None,
                        names=["lon", "lat", "t"])
    nodes = pd.read_csv(folder / f"{trip_id}.nodes", sep=r"\s+", header=None,
                        names=["lon", "lat"])
    arcs = pd.read_csv(folder / f"{trip_id}.arcs", sep=r"\s+", header=None,
                       names=["from_node", "to_node"], dtype=int)

    route_file = folder / f"{trip_id}.route"
    route = None
    if route_file.exists():
        route = np.loadtxt(route_file, dtype=int, ndmin=1)

    return {"trip_id": trip_id, "track": track, "nodes": nodes,
            "arcs": arcs, "route": route}


def to_xy(lon, lat, lon0, lat0):
    """
    Convert longitude/latitude (degrees) to x/y in METRES around a reference
    point (lon0, lat0) using a local equirectangular projection.
    Good enough for a single city-sized trip (error well under 1% at this scale).
    """
    x = np.radians(np.asarray(lon) - lon0) * EARTH_RADIUS_M * np.cos(np.radians(lat0))
    y = np.radians(np.asarray(lat) - lat0) * EARTH_RADIUS_M
    return x, y
