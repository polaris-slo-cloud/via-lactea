#!/usr/bin/env python3
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set, Iterable
import math
import heapq
import random

from . import config
from . import profiles

# -------------------------
# Data structures
# -------------------------

@dataclass(frozen=True)
class Node:
    nid: str
    kind: str  # "sat" | "cloud" | "edge"

@dataclass
class Edge:
    u: str
    v: str
    kind: str   # "isl" | "downlink" | "edge_backhaul"
    bw_range: Tuple[float, float]   # (lo, hi) Mbps
    prop_range: Tuple[float, float] # (lo, hi) ms

class Topology:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.adj: Dict[str, List[Edge]] = {}

    def add_node(self, node: Node):
        if node.nid not in self.nodes:
            self.nodes[node.nid] = node
            self.adj[node.nid] = []

    def add_undirected_edge(
        self,
        u: str,
        v: str,
        kind: str,
        bw_range: Tuple[float, float],
        prop_range: Tuple[float, float],
    ):
        e1 = Edge(u=u, v=v, kind=kind, bw_range=bw_range, prop_range=prop_range)
        e2 = Edge(u=v, v=u, kind=kind, bw_range=bw_range, prop_range=prop_range)
        self.adj[u].append(e1)
        self.adj[v].append(e2)

    def neighbors(self, u: str) -> List[Edge]:
        return self.adj.get(u, [])

# -------------------------
# Link ranges (from config)
# -------------------------

def _link_ranges(kind: str) -> Dict[str, Tuple[float, float]]:
    lr = getattr(config, "LINK_RANGES", None)
    if not lr:
        lr = {
            "isl":           {"bw": (1000000.0, 1000000.0), "prop": (30.0, 40.0)},
            "downlink":      {"bw": (300.0, 300.0),         "prop": (50.0, 60.0)},
            "edge_backhaul": {"bw": (100.0, 100.0),         "prop": (25.0, 35.0)},
            "default":       {"bw": (10.0, 10.0),           "prop": (100.0, 100.0)},
        }
        setattr(config, "LINK_RANGES", lr)
    return lr.get(kind, lr["default"])

# -------------------------
# Builders
# -------------------------

def build_topology_from_json(path: str = "resources/topo_250_new.json") -> Topology:
    """
    Build a topology from a Stardust JSON snapshot.

    Expects a structure with fields:
      - Satellites: [{ "Name": "STARLINK-1008", ... }, ...]
      - Grounds:    [{ "Name": "Vienna", ... }, ...]
      - Links:      [{ "NodeName1": "...", "NodeName2": "..." }, ...]
      - States:     [ { "Time": ..., "NodeStates": [...] }, ... ]

    Uses the first state in States and only the links marked as Established there.
    """
    with open(path, "r") as f:
        data = json.load(f)

    topo = Topology()

    for sat in data.get("Satellites", []):
        topo.add_node(Node(nid=sat["Name"], kind="sat"))

    for gs in data.get("Grounds", []):
        topo.add_node(Node(nid=gs["Name"], kind="edge"))

    sat_count = len(data.get("Satellites", []))
    edge_count = len(data.get("Grounds", []))

    config.NODE_COUNTS = {"sat": sat_count, "edge": edge_count, "cloud": 1}

    links = data.get("Links", [])
    states = data.get("States", [])
    if not states:
        return topo

    state = states[0]

    active_link_indices: Set[int] = set()
    for ns in state.get("NodeStates", []):
        for idx in ns.get("Established", []):
            active_link_indices.add(idx)

    isl_lr = _link_ranges("isl")
    dl_lr = _link_ranges("downlink")

    added_pairs: Set[Tuple[str, str]] = set()

    for idx in active_link_indices:
        if idx < 0 or idx >= len(links):
            continue
        link = links[idx]
        u = link["NodeName1"]
        v = link["NodeName2"]
        if u == v:
            continue

        key = (u, v) if u <= v else (v, u)
        if key in added_pairs:
            continue
        added_pairs.add(key)

        if u.startswith("STARLINK") and v.startswith("STARLINK"):
            kind = "isl"
            bw_range = isl_lr["bw"]
            prop_range = isl_lr["prop"]
        else:
            kind = "downlink"
            bw_range = dl_lr["bw"]
            prop_range = dl_lr["prop"]

        topo.add_undirected_edge(u, v, kind, bw_range, prop_range)

    return topo


def build_topology(
    sats_per_ring: int,
    num_rings: int,
    cloud_count: int,
    edge_count: int,
    *,
    isl_neighbor_span: int = 1,
    gateways_per_ring: int = 2,
    inter_ring_links: bool = True,
) -> Topology:
    """
    Build a concrete multi-ring satellite topology + cloud + edge gateways.

    sat IDs:   "sat_r{r}_i{i}"
    cloud IDs: "cloud{k}"
    edge IDs:  "edge{k}"
    """
    topo = Topology()

    for r in range(num_rings):
        for i in range(sats_per_ring):
            topo.add_node(Node(nid=f"sat_r{r}_i{i}", kind="sat"))
    for k in range(cloud_count):
        topo.add_node(Node(nid=f"cloud{k}", kind="cloud"))
    for k in range(edge_count):
        topo.add_node(Node(nid=f"edge{k}", kind="edge"))

    isl_lr = _link_ranges("isl")
    dl_lr  = _link_ranges("downlink")
    eb_lr  = _link_ranges("edge_backhaul")

    for r in range(num_rings):
        for i in range(sats_per_ring):
            u = f"sat_r{r}_i{i}"
            for d in range(1, isl_neighbor_span + 1):
                v1 = f"sat_r{r}_i{(i + d) % sats_per_ring}"
                v2 = f"sat_r{r}_i{(i - d) % sats_per_ring}"
                topo.add_undirected_edge(u, v1, "isl", isl_lr["bw"], isl_lr["prop"])
                topo.add_undirected_edge(u, v2, "isl", isl_lr["bw"], isl_lr["prop"])

    if inter_ring_links and num_rings > 1:
        for r in range(num_rings - 1):
            for i in range(sats_per_ring):
                u = f"sat_r{r}_i{i}"
                v = f"sat_r{r+1}_i{i}"
                topo.add_undirected_edge(u, v, "isl", isl_lr["bw"], isl_lr["prop"])

    for r in range(num_rings):
        step = max(1, sats_per_ring // max(1, gateways_per_ring))
        gateway_indices = [(g * step) % sats_per_ring for g in range(gateways_per_ring)]
        for gi in gateway_indices:
            s = f"sat_r{r}_i{gi}"
            for k in range(cloud_count):
                c = f"cloud{k}"
                topo.add_undirected_edge(s, c, "downlink", dl_lr["bw"], dl_lr["prop"])

    for k in range(edge_count):
        e = f"edge{k}"
        c = f"cloud{k % cloud_count}" if cloud_count > 0 else None
        if c:
            topo.add_undirected_edge(e, c, "edge_backhaul", eb_lr["bw"], eb_lr["prop"])

    return topo

# -------------------------
# Deterministic SLO Dijkstra (prop-only)
# -------------------------

def _edge_prop_mid(edge: Edge) -> float:
    lo, hi = edge.prop_range
    return 0.5 * (float(lo) + float(hi))

def _edge_available(edge: Edge) -> bool:
    lo, hi = edge.bw_range
    return float(hi) > 0.0

def dijkstra_base_path_between_nodes(
    topo: Topology,
    src: str,
    dst: str,
    slo_ms: Optional[float] = None,
) -> Tuple[float, List[str]]:
    """
    Shortest path (propagation-only). Edges with bw_hi <= 0 are skipped.
    If slo_ms is set, we drop ONLY edges whose individual propagation > slo_ms.
    No cumulative/path-level pruning.
    Returns (latency_ms, path_nodes). If unreachable: (math.inf, []).
    """
    if src == dst:
        return 0.0, [src]

    if src not in topo.nodes or dst not in topo.nodes:
        return math.inf, []

    dist: Dict[str, float] = {nid: math.inf for nid in topo.nodes}
    prev: Dict[str, str] = {}
    seen: Set[str] = set()

    dist[src] = 0.0
    pq: List[Tuple[float, str]] = [(0.0, src)]

    while pq:
        d, u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)

        if u == dst:
            path = [dst]
            while path[-1] != src:
                p = prev.get(path[-1])
                if p is None:
                    return math.inf, []
                path.append(p)
            path.reverse()
            return d, path

        for e in topo.neighbors(u):
            if not _edge_available(e):
                continue
            edge_cost = _edge_prop_mid(e)

            if (slo_ms is not None) and (edge_cost > float(slo_ms)):
                continue

            w = d + edge_cost
            if w < dist[e.v]:
                dist[e.v] = w
                prev[e.v] = u
                heapq.heappush(pq, (w, e.v))

    return math.inf, []

# -------------------------
# Runtime Dijkstra (payload-aware)
# -------------------------

_BW_EPS_Mbps = float(getattr(config, "BW_EPS_Mbps", 1e-6))
if (not math.isfinite(_BW_EPS_Mbps)) or _BW_EPS_Mbps <= 0.0:
    _BW_EPS_Mbps = 1e-6

_HUGE_HOP_MS = float(getattr(config, "HUGE_HOP_MS", 1e12))
if (not math.isfinite(_HUGE_HOP_MS)) or _HUGE_HOP_MS <= 0.0:
    _HUGE_HOP_MS = 1e12

def _sample_link_state(edge: Edge, rng: random.Random) -> Tuple[float, float]:
    """
    Sample bandwidth and propagation.

    Critical:
    - clamp bandwidth away from 0 to prevent inf explosions on long paths
    - clamp propagation to >= 0 and finite
    """
    lo_bw = float(edge.bw_range[0])
    hi_bw = float(edge.bw_range[1])
    lo_p  = float(edge.prop_range[0])
    hi_p  = float(edge.prop_range[1])

    bw = rng.uniform(lo_bw, hi_bw)
    prop = rng.uniform(lo_p, hi_p)

    if (bw is None) or (not math.isfinite(float(bw))) or float(bw) <= 0.0:
        bw = _BW_EPS_Mbps
    if (prop is None) or (not math.isfinite(float(prop))) or float(prop) < 0.0:
        prop = 0.0

    return float(bw), float(prop)

def _hop_latency_ms(edge: Edge, payload_mb: float, rng: random.Random) -> float:
    """
    Compute per-hop latency.

    Important:
    - never return inf/nan, because that poisons aggregation (NaN means empty slice)
    - if something goes wrong, return a large finite penalty
    """
    bw_mbps, prop_ms = _sample_link_state(edge, rng)

    # bw_mbps is clamped in _sample_link_state
    tx_ms = (float(payload_mb) / (float(bw_mbps) / 8.0)) * 1000.0

    j_lo, j_hi = getattr(config, "HOP_JITTER_MS", (0.0, 10.0))
    jitter = rng.uniform(float(j_lo), float(j_hi))

    out = float(tx_ms) + float(prop_ms) + float(jitter)
    if not math.isfinite(out):
        return _HUGE_HOP_MS
    if out < 0.0:
        return 0.0
    return out

def dijkstra_latency_between_nodes(
    topo: Topology, src: str, dst: str, payload_mb: float, rng: random.Random
) -> float:
    """Runtime shortest path with payload-aware edge costs."""
    if src == dst:
        return 0.0
    if src not in topo.nodes or dst not in topo.nodes:
        return float("inf")

    dist: Dict[str, float] = {nid: float("inf") for nid in topo.nodes}
    dist[src] = 0.0
    pq: List[Tuple[float, str]] = [(0.0, src)]
    seen: Set[str] = set()

    while pq:
        d, u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        if u == dst:
            return d
        for e in topo.neighbors(u):
            if not _edge_available(e):
                continue
            cost = _hop_latency_ms(e, float(payload_mb), rng)
            if not math.isfinite(cost):
                cost = _HUGE_HOP_MS
            nd = d + cost
            if nd < dist[e.v]:
                dist[e.v] = nd
                heapq.heappush(pq, (nd, e.v))
    return float("inf")

# -------------------------
# Route materialization & hops
# -------------------------

def _dijkstra_path(topo: Topology, src: str, dst: str, weight_fn) -> Optional[List[str]]:
    """Generic Dijkstra that also reconstructs path of node IDs using weight_fn(edge)->cost."""
    if src == dst:
        return [src]
    if src not in topo.nodes or dst not in topo.nodes:
        return None

    dist: Dict[str, float] = {nid: float("inf") for nid in topo.nodes}
    prev: Dict[str, Optional[str]] = {nid: None for nid in topo.nodes}
    dist[src] = 0.0

    pq: List[Tuple[float, str]] = [(0.0, src)]
    seen: Set[str] = set()

    while pq:
        d, u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        if u == dst:
            break
        for e in topo.neighbors(u):
            w = weight_fn(e)
            if not math.isfinite(w):
                continue
            nd = d + w
            if nd < dist[e.v]:
                dist[e.v] = nd
                prev[e.v] = u
                heapq.heappush(pq, (nd, e.v))

    if not math.isfinite(dist[dst]):
        return None

    # reconstruct safely
    path: List[str] = []
    cur: Optional[str] = dst
    visited: Set[str] = set()
    while cur is not None:
        if cur in visited:
            return None
        visited.add(cur)
        path.append(cur)
        if cur == src:
            break
        cur = prev[cur]

    if not path or path[-1] != src:
        return None

    path.reverse()
    return path

def route_hops_nodes(topo: Topology, src: str, dst: str, rng: Optional[random.Random] = None) -> List[Edge]:
    """
    Materialize the hop sequence (as Edge objects) along the deterministic base path.
    (You can switch to runtime weight if you want stochastic paths.)
    """
    def base_weight(e: Edge) -> float:
        if not _edge_available(e):
            return float("inf")
        return _edge_prop_mid(e)

    path = _dijkstra_path(topo, src, dst, base_weight)
    if path is None or len(path) < 2:
        return []

    hops: List[Edge] = []
    for u, v in zip(path[:-1], path[1:]):
        found = False
        for e in topo.neighbors(u):
            if e.v == v:
                hops.append(e)
                found = True
                break
        if not found:
            return []
    return hops

def expected_hops_between_nodes(topo: Topology, src: str, dst: str) -> int:
    """Hop count along the deterministic base path; 0 if no path."""
    path = route_hops_nodes(topo, src, dst)
    return len(path)

# -------------------------
# Compute & payload (unchanged)
# -------------------------

def _sample_runtime_lognormal(mean_ms: float, cv: float, rng: random.Random) -> float:
    if mean_ms <= 0.0:
        return 0.0
    sigma2 = math.log(1.0 + cv * cv)
    sigma = math.sqrt(sigma2)
    mu = math.log(mean_ms) - 0.5 * sigma2
    return rng.lognormvariate(mu, sigma)

def compute_time_ms(module: str, node: Node, rng: random.Random, task_profile_name: str) -> float:
    prof = profiles.TASK_PROFILES.get(task_profile_name, profiles.TASK_PROFILES["object-det"])
    is_prefix = module.startswith("resnet_")
    bucket = "prefix" if is_prefix else "suffix"
    base_mean = prof[node.kind][bucket]
    cv = getattr(config, "CV_PREFIX", 0.15) if is_prefix else getattr(config, "CV_SUFFIX", 0.15)
    return _sample_runtime_lognormal(float(base_mean), float(cv), rng)

def module_output_mb(module: str) -> float:
    return float(profiles.OUTPUT_SIZES_MB.get(module, 0.0))

# ---------- helper: make a filtered topology view ----------

def _filtered_topology_view(topo: Topology, per_edge_prop_cap_ms: Optional[float]) -> Topology:
    """
    Create a filtered copy of `topo` that:
      - keeps all nodes
      - keeps only edges with available bandwidth (bw_hi > 0)
      - if per_edge_prop_cap_ms is not None: also requires deterministic propagation(mid) <= cap
    This ensures SLO feasibility is enforced at the edge level before path search.
    """
    ft = Topology()
    for _nid, node in topo.nodes.items():
        ft.add_node(Node(nid=node.nid, kind=node.kind))

    seen_pairs: Set[Tuple[str, str]] = set()
    for _u, edges in topo.adj.items():
        for e in edges:
            key = (e.u, e.v) if e.u <= e.v else (e.v, e.u)
            if key in seen_pairs:
                continue
            if not _edge_available(e):
                continue
            if per_edge_prop_cap_ms is not None and _edge_prop_mid(e) > float(per_edge_prop_cap_ms):
                continue
            ft.add_undirected_edge(e.u, e.v, e.kind, e.bw_range, e.prop_range)
            seen_pairs.add(key)
    return ft