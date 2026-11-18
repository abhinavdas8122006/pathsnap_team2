"""
hmm.py
Hidden Markov Model + Viterbi route reconstruction.

Hidden states   : for each GPS point, its candidate road segments (arcs).
Observations    : the GPS points.
Emission prob.  : the classifier's probability that the candidate is the correct road.
Transition prob.: how plausible it is to drive from candidate A (at point t-1) to candidate B (at point t).
                  We compare the DRIVING distance along the road network with the STRAIGHT-LINE
                  distance between the two GPS points:  P ~ exp(-|d_network - d_gps| / beta).
                  A car 1 second later is usually on the same road or the next connected road,
                  so jumps to unconnected roads get a very low probability.
Viterbi         : finds the single most likely sequence of roads for the whole trip.
"""
import heapq
import numpy as np

DEFAULTS = {
    "top_k": 8,              # keep the K most probable candidates per GPS point
    "beta": 10.0,            # metres; larger = more tolerant to mismatch between network and GPS distance
    "emission_weight": 1.0,  # how much we trust the classifier vs. the road-network transition
    "gap_seconds": 30.0,     # a time gap larger than this breaks the chain (GPS was off)
    "unreachable_logp": -30.0,  # log-prob for transitions with no path (keeps decoding alive on disconnected roads)
}


class RoadGraph:
    """Directed road network with cached, distance-limited Dijkstra searches."""

    def __init__(self, g):
        self.from_node, self.to_node = g["from_node"], g["to_node"]
        self.length = np.hypot(g["bx"] - g["ax"], g["by"] - g["ay"])
        n_nodes = int(max(self.from_node.max(), self.to_node.max())) + 1
        order = np.argsort(self.from_node, kind="stable")
        self.out_arcs = order
        self.out_start = np.searchsorted(self.from_node[order], np.arange(n_nodes + 1))
        self.cache = {}

    def distances_from(self, node, limit):
        """Shortest driving distance (m) from `node` to every node reachable within `limit` metres."""
        hit = self.cache.get(node)
        if hit is not None and hit[0] >= limit:
            return hit[1]
        dist = {node: 0.0}
        heap = [(0.0, node)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, np.inf) or d > limit:
                continue
            for arc in self.out_arcs[self.out_start[u]:self.out_start[u + 1]]:
                v = self.to_node[arc]
                nd = d + self.length[arc]
                if nd <= limit and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        self.cache[node] = (limit, dist)
        return dist


def viterbi_decode(df, proba, g, params=None):
    """
    df    : candidate rows of ONE trip (columns point, arc_id, proj_frac)
    proba : classifier probability for each row
    g     : geometry dict from features.prepare_geometry
    Returns an array with the chosen arc_id for every GPS point (index = point number).
    """
    p = {**DEFAULTS, **(params or {})}
    graph = RoadGraph(g)
    T = len(g["px"])

    # ---- candidate sets per point (top-K by classifier probability)
    work = df[["point", "arc_id", "proj_frac"]].copy()
    work["p"] = proba
    work = work.sort_values(["point", "p"], ascending=[True, False])
    work = work.groupby("point").head(p["top_k"])
    groups = {pt: grp for pt, grp in work.groupby("point")}
    points = np.array(sorted(groups))

    eps = 1e-6
    prev_states = prev_score = None
    back = {}
    for idx, pt in enumerate(points):
        grp = groups[pt]
        arcs = grp["arc_id"].values
        frac = grp["proj_frac"].values
        emis = p["emission_weight"] * np.log(np.clip(grp["p"].values, eps, 1.0))

        if prev_states is None:
            score = emis
            back[pt] = None
        else:
            prev_pt, p_arcs, p_frac = prev_states
            dt = g["t"][pt] - g["t"][prev_pt]
            if dt > p["gap_seconds"]:
                # GPS gap: no reliable motion information -> restart from emissions only
                best_prev = int(np.argmax(prev_score))
                score = prev_score[best_prev] + emis
                back[pt] = np.full(len(arcs), best_prev)
            else:
                gps_d = np.hypot(g["px"][pt] - g["px"][prev_pt], g["py"][pt] - g["py"][prev_pt])
                limit = max(3.0 * gps_d + 100.0, 200.0)
                trans = np.full((len(p_arcs), len(arcs)), p["unreachable_logp"])
                for i, (a, ua) in enumerate(zip(p_arcs, p_frac)):
                    len_a = graph.length[a]
                    # same road: distance along the road
                    same = arcs == a
                    if same.any():
                        # signed distance along the road: moving BACKWARDS is penalised,
                        # which stops the decoder from choosing the opposite-direction twin road
                        along = (frac[same] - ua) * len_a
                        trans[i, same] = -np.abs(along - gps_d) / p["beta"]
                    # different road: rest of road a + shortest path + start of road b
                    dist = graph.distances_from(graph.to_node[a], limit)
                    other = np.where(~same)[0]
                    if len(other):
                        sp = np.array([dist.get(graph.from_node[b], np.inf) for b in arcs[other]])
                        net = (1 - ua) * len_a + sp + frac[other] * graph.length[arcs[other]]
                        ok = np.isfinite(net)
                        trans[i, other[ok]] = -np.abs(net[ok] - gps_d) / p["beta"]
                total = prev_score[:, None] + trans
                back[pt] = np.argmax(total, axis=0)
                score = total[back[pt], np.arange(len(arcs))] + emis
        prev_states = (pt, arcs, frac)
        prev_score = score
        groups[pt] = (arcs, frac)

    # ---- trace back the best path
    chosen = np.full(T, -1, dtype=np.int64)
    j = int(np.argmax(prev_score))
    for idx in range(len(points) - 1, -1, -1):
        pt = points[idx]
        arcs, _ = groups[pt]
        chosen[pt] = arcs[j]
        if back[pt] is not None:
            j = int(back[pt][j])
    return chosen


def route_from_points(chosen):
    """Collapse the per-point arcs into the driven route (consecutive duplicates removed)."""
    arcs = chosen[chosen >= 0]
    if len(arcs) == 0:
        return arcs
    keep = np.r_[True, arcs[1:] != arcs[:-1]]
    return arcs[keep]
