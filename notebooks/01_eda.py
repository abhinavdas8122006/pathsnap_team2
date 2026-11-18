# %% [markdown]
# # 01 - Dataset exploration (EDA)
# Partitions used: TRAIN (80 trips) and STUDENT_TEST (10 trips).

# %% Setup: make the src folder importable
import sys
from pathlib import Path
ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from data_loader import list_trips, load_trip

# %% 1. Look at one trip's raw tables
trip = load_trip("train", "00000001")
print(trip["track"].head())
print(trip["nodes"].head())
print(trip["arcs"].head())
print("First route arcs:", trip["route"][:10])

# %% 2. Dataset inventory: one row per trip
rows = []
for partition in ["train", "student_test"]:
    for tid in list_trips(partition):
        tr = load_trip(partition, tid)
        track, arcs, nodes, route = tr["track"], tr["arcs"], tr["nodes"], tr["route"]
        dt = np.diff(track["t"].values)

        # Road-network connectivity: treat roads as undirected links between nodes
        n = len(nodes)
        graph = coo_matrix((np.ones(len(arcs)), (arcs.from_node, arcs.to_node)), shape=(n, n))
        n_components, _ = connected_components(graph, directed=False)
        used_nodes = np.union1d(arcs.from_node.unique(), arcs.to_node.unique())

        # Is the reference route connected? (end node of arc i == start node of arc i+1)
        if route is not None and len(route) > 1:
            route_connected = np.mean(arcs.to_node.values[route[:-1]] == arcs.from_node.values[route[1:]])
        else:
            route_connected = np.nan

        rows.append({
            "partition": partition, "trip_id": tid,
            "n_gps_points": len(track),
            "n_nodes": n, "n_arcs": len(arcs),
            "n_route_arcs": 0 if route is None else len(route),
            "duration_s": track["t"].iloc[-1] - track["t"].iloc[0],
            "missing_values": int(track.isna().sum().sum()),
            "non_increasing_time": int((dt <= 0).sum()),
            "share_1s_steps": np.mean(dt == 1),
            "max_time_gap_s": dt.max(),
            "gaps_over_10s": int((dt > 10).sum()),
            "isolated_nodes": n - len(used_nodes),
            "network_components": n_components,
            "route_connected_share": route_connected,
            "center_lon": track["lon"].mean(), "center_lat": track["lat"].mean(),
        })

inventory = pd.DataFrame(rows)
inventory.to_csv(OUT / "eda_inventory.csv", index=False)
print(inventory.groupby("partition").size())
inventory.head(10)

# %% 3. Summary statistics
summary = inventory.groupby("partition")[[
    "n_gps_points", "n_arcs", "n_route_arcs", "duration_s",
    "max_time_gap_s", "gaps_over_10s", "missing_values", "network_components"]].describe().T
summary.to_csv(OUT / "eda_summary.csv")
print(summary)

# %% 4. Anomalies: trips with big GPS gaps
print(inventory.sort_values("max_time_gap_s", ascending=False)
      [["partition", "trip_id", "max_time_gap_s", "gaps_over_10s", "share_1s_steps"]].head(10))

# %% 5. Plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
inventory[inventory.partition == "train"]["n_gps_points"].plot.hist(bins=30, ax=axes[0])
axes[0].set_title("GPS points per trip (train)")
axes[0].set_xlabel("points")

all_dt = []
for tid in list_trips("train"):
    all_dt.append(np.diff(load_trip("train", tid)["track"]["t"].values))
all_dt = np.concatenate(all_dt)
axes[1].hist(np.clip(all_dt, 0, 20), bins=40, log=True)
axes[1].set_title("Time between GPS points (train, clipped at 20 s)")
axes[1].set_xlabel("seconds")

for part, colour in [("train", "tab:blue"), ("student_test", "tab:orange")]:
    sub = inventory[inventory.partition == part]
    axes[2].scatter(sub.center_lon, sub.center_lat, s=12, c=colour, label=part)
axes[2].set_title("Where the trips are (trip centre)")
axes[2].set_xlabel("longitude")
axes[2].set_ylabel("latitude")
axes[2].legend()
plt.tight_layout()
plt.savefig(OUT / "eda_overview.png", dpi=120)
plt.show()

# %% 6. Plot one trajectory over its road network
trip = load_trip("train", "00000003")
track, nodes, arcs, route = trip["track"], trip["nodes"], trip["arcs"], trip["route"]
fig, ax = plt.subplots(figsize=(8, 8))
pad = 0.005
in_box = ((nodes.lon.between(track.lon.min() - pad, track.lon.max() + pad)) &
          (nodes.lat.between(track.lat.min() - pad, track.lat.max() + pad))).values
for a, b in arcs[in_box[arcs.from_node] & in_box[arcs.to_node]].values:
    ax.plot(nodes.lon[[a, b]], nodes.lat[[a, b]], color="lightgrey", lw=0.6)
for arc_id in route:
    a, b = arcs.iloc[arc_id]
    ax.plot(nodes.lon[[a, b]], nodes.lat[[a, b]], color="green", lw=2.5)
ax.scatter(track.lon, track.lat, s=3, color="blue", label="GPS")
ax.set_title("Trip 00000003 (train): grey = roads, green = reference route, blue = GPS")
plt.savefig(OUT / "eda_trip_00000003.png", dpi=120)
plt.show()

# %% 7. Interactive Folium map (mandatory for the report)
import folium

def folium_trip_map(trip, filename):
    track, nodes, arcs, route = trip["track"], trip["nodes"], trip["arcs"], trip["route"]
    m = folium.Map(location=[track.lat.mean(), track.lon.mean()], zoom_start=15,
                   tiles="OpenStreetMap")

    # Layer 1: nearby road network (only roads inside the trip's bounding box)
    roads = folium.FeatureGroup(name="Road network (nearby)")
    pad = 0.003
    in_box = ((nodes.lon.between(track.lon.min() - pad, track.lon.max() + pad)) &
              (nodes.lat.between(track.lat.min() - pad, track.lat.max() + pad))).values
    for a, b in arcs[in_box[arcs.from_node] & in_box[arcs.to_node]].values[:6000]:
        folium.PolyLine([(nodes.lat[a], nodes.lon[a]), (nodes.lat[b], nodes.lon[b])],
                        color="grey", weight=1, opacity=0.5).add_to(roads)
    roads.add_to(m)

    # Layer 2: reference route (green)
    if route is not None:
        ref = folium.FeatureGroup(name="Reference route (green)")
        for arc_id in route:
            a, b = arcs.iloc[arc_id]
            folium.PolyLine([(nodes.lat[a], nodes.lon[a]), (nodes.lat[b], nodes.lon[b])],
                            color="green", weight=5, opacity=0.8).add_to(ref)
        ref.add_to(m)

    # Layer 3: raw GPS points (blue dots), every 2nd point to keep the file small
    gps = folium.FeatureGroup(name="Raw GPS points (blue)")
    for _, p in track.iloc[::2].iterrows():
        folium.CircleMarker((p.lat, p.lon), radius=2, color="blue", fill=True,
                            tooltip=f"t={p.t:.0f}s").add_to(gps)
    gps.add_to(m)

    folium.Marker((track.lat.iloc[0], track.lon.iloc[0]), tooltip="Start",
                  icon=folium.Icon(color="green")).add_to(m)
    folium.Marker((track.lat.iloc[-1], track.lon.iloc[-1]), tooltip="End",
                  icon=folium.Icon(color="red")).add_to(m)
    folium.LayerControl().add_to(m)
    m.save(str(OUT / filename))
    return m

folium_trip_map(load_trip("train", "00000003"), "map_eda_train_00000003.html")
