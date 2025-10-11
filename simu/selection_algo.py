# selection_helper.py
# --- Fast SSSP helpers (connectivity-probing build) ---------------------------

from typing import Dict, Tuple, Optional, List, Iterable, Any
import math
import heapq
import os
import re
import time
from collections import OrderedDict
from collections.abc import Iterable as _IterableABC

from . import config

NodeID = Any  # preserve whatever type your topology uses (str/int/objects)

# ==== DEBUG / probe toggles (cheap) ===========================================
DEBUG = False
MIN_EDGE_MS = 0.0
SHOW_PROBE = False
PROBE_NODES = 50
PROBE_NEIGHBORS = 50
# ==============================================================================

# Collapse latency ranges: "min" | "mid" | "max"
WEIGHT_FROM_RANGE = "mid"

# ---- SSSP algo toggle: "auto" (default), "binary", or "radix"
SSSP_ALGO = os.getenv("VL_SSSP_ALGO", "auto").strip().lower()

# ---- When to switch from pairwise to full SSSP tree per (src,slo) ------------
_SSSP_TREE_THRESHOLD = int(os.getenv("VL_SSSP_TREE_THRESHOLD", "3"))

# ---- Global per-source tree cache (LRU+TTL) ----------------------------------
_GLOBAL_SSSP_LRU_CAP = int(os.getenv("VL_SSSP_LRU_CAP", "128"))
_GLOBAL_SSSP_TTL_SEC = float(os.getenv("VL_SSSP_TTL_SEC", "60"))

class _TreeEntry:
    __slots__ = ("dist", "parent", "tstamp")
    def __init__(self, dist, parent, tstamp):
        self.dist = dist
        self.parent = parent
        self.tstamp = tstamp

def set_sssp_algo(name: str):
    """Change algorithm choice at runtime: 'binary' | 'radix' | 'auto'."""
    global SSSP_ALGO
    SSSP_ALGO = (name or "auto").strip().lower()

def _dbg(*args):
    if DEBUG:
        print("[vl_selection]", *args)

# ------------------------------------------------------------------------------
# Per-process resolver cache
_RESOLVER_CACHE: Dict[int, Dict[Any, Any]] = {}

def _ensure_resolver(ftopo) -> Dict[Any, Any]:
    key = id(ftopo)
    if key in _RESOLVER_CACHE:
        return _RESOLVER_CACHE[key]

    by_id: Dict[Any, Any] = {}
    nodes_iter = None

    if hasattr(ftopo, "nodes"):
        nodes_attr = getattr(ftopo, "nodes")
        nodes_iter = nodes_attr() if callable(nodes_attr) else nodes_attr
    if nodes_iter is None and hasattr(ftopo, "adj"):
        try:
            nodes_iter = getattr(ftopo, "adj").keys()
        except Exception:
            pass
    if nodes_iter is None and hasattr(ftopo, "graph"):
        g = getattr(ftopo, "graph")
        if isinstance(g, dict):
            nodes_iter = g.keys()

    if nodes_iter is not None:
        for obj in nodes_iter:
            by_id[obj] = obj
            nid = getattr(obj, "nid", None)
            if nid is not None:
                by_id[nid] = obj
                by_id[str(nid)] = obj
            else:
                by_id[str(obj)] = obj

    for m in ("get_node", "node", "getVertex", "vertex"):
        if hasattr(ftopo, m) and callable(getattr(ftopo, m)):
            by_id["_fetcher_"] = getattr(ftopo, m)
            break

    _RESOLVER_CACHE[key] = by_id
    return by_id

# --- Neighbor normalization ----------------------------------------------------

_WEIGHT_FIELDS = (
    "prop", "prop_ms",
    "latency_ms", "lat_ms", "delay_ms",
    "latency", "delay",
    "weight", "w", "cost", "time", "ms"
)
_RANGE_FIELDS = ("prop_range", "lat_range", "delay_range")

_DST_FIELDS = (
    "dst", "to", "v", "neighbor", "nbr", "node", "target",
    "id", "nid", "dst_id", "to_id"
)

def _call_if_callable(x):
    return x() if callable(x) else x

def _collapse_range(rng) -> Optional[float]:
    try:
        a, b = float(rng[0]), float(rng[1])
        if WEIGHT_FROM_RANGE == "min":
            return a
        if WEIGHT_FROM_RANGE == "max":
            return b
        return 0.5 * (a + b)
    except Exception:
        return None

def _edge_to_pair(e) -> Optional[Tuple[NodeID, float]]:
    # Tuple-like (dst, w)
    if isinstance(e, tuple) and len(e) >= 2:
        v, w = e[0], e[1]
        v = getattr(v, "nid", v)
        try:
            return (v, float(_call_if_callable(w)))
        except Exception:
            return None

    # Object-like
    dst = None
    for name in _DST_FIELDS:
        if hasattr(e, name):
            dst = _call_if_callable(getattr(e, name))
            break
    if dst is not None and hasattr(dst, "nid"):
        dst = getattr(dst, "nid")

    w = None
    for name in _WEIGHT_FIELDS:
        if hasattr(e, name):
            w = _call_if_callable(getattr(e, name))
            break

    if w is None:
        for rname in _RANGE_FIELDS:
            if hasattr(e, rname):
                rng = _call_if_callable(getattr(e, rname))
                w = _collapse_range(rng)
                break

    if dst is None or w is None:
        return None

    try:
        return (dst, float(w))
    except Exception:
        return None

# --- Ring-forward filtering helpers -------------------------------------------

_RING_ATTR_CANDIDATES = (
    "ring_id", "plane_id", "plane", "orbit_id", "orbit", "ring", "planeIdx", "planeIndex"
)
_INDEX_ATTR_CANDIDATES = (
    "idx", "index", "sat_idx", "slot", "ordinal", "satIndex", "satId"
)

def _get_node_obj(ftopo, x):
    resolver = _ensure_resolver(ftopo)
    if x in resolver:
        return resolver[x]
    fetcher = resolver.get("_fetcher_")
    if fetcher:
        try:
            return fetcher(x)
        except Exception:
            return x
    return x

def _extract_ring_idx(ftopo, node_id_or_obj) -> Tuple[Optional[int], Optional[int]]:
    nobj = _get_node_obj(ftopo, node_id_or_obj)
    ring_id = None
    sat_idx = None

    for a in _RING_ATTR_CANDIDATES:
        if hasattr(nobj, a):
            try:
                ring_id = int(getattr(nobj, a)); break
            except Exception:
                pass
    for a in _INDEX_ATTR_CANDIDATES:
        if hasattr(nobj, a):
            try:
                sat_idx = int(getattr(nobj, a)); break
            except Exception:
                pass

    if (ring_id is None or sat_idx is None) and hasattr(nobj, "nid"):
        nid = getattr(nobj, "nid")
        for a in _RING_ATTR_CANDIDATES:
            if hasattr(nid, a):
                try:
                    ring_id = int(getattr(nid, a)); break
                except Exception:
                    pass
        for a in _INDEX_ATTR_CANDIDATES:
            if hasattr(nid, a):
                try:
                    sat_idx = int(getattr(nid, a)); break
                except Exception:
                    pass

    if (ring_id is None or sat_idx is None):
        s = str(getattr(nobj, "nid", nobj))
        m = re.search(r"[Rr]ing[-_]?(\d+).*?[Ss](?:at)?[-_]?(\d+)", s)
        if m:
            try:
                if ring_id is None: ring_id = int(m.group(1))
                if sat_idx is None: sat_idx = int(m.group(2))
            except Exception:
                pass

    return ring_id, sat_idx

def _is_future_neighbor(ftopo, u_id, v_id) -> bool:
    ru, iu = _extract_ring_idx(ftopo, u_id)
    rv, iv = _extract_ring_idx(ftopo, v_id)
    if ru is None or rv is None or iu is None or iv is None:
        return True
    if ru != rv:
        return True
    return iv > iu

# --- Fast adjacency snapshot ---------------------------------------------------

def _ensure_adj_snapshot(ftopo):
    """Create a compact, read-only adjacency cache on the filtered topology."""
    if getattr(ftopo, "_vl_adj_ready", False):
        return
    adj: Dict[Any, List[Tuple[Any, float]]] = {}

    nodes_iter = None
    if hasattr(ftopo, "nodes"):
        nodes_attr = getattr(ftopo, "nodes")
        nodes_iter = nodes_attr() if callable(nodes_attr) else nodes_attr
    if nodes_iter is None and hasattr(ftopo, "adj"):
        try:
            nodes_iter = getattr(ftopo, "adj").keys()
        except Exception:
            pass
    if nodes_iter is None and hasattr(ftopo, "graph"):
        g = getattr(ftopo, "graph")
        if isinstance(g, dict):
            nodes_iter = g.keys()
    if nodes_iter is None:
        return

    for u in nodes_iter:
        u_id = getattr(u, "nid", u)
        vs = _neighbors_as_pairs_no_snapshot(ftopo, u_id)
        if vs:
            adj[u_id] = vs

    setattr(ftopo, "_vl_adj", adj)
    setattr(ftopo, "_vl_adj_ready", True)

def _neighbors_raw(ftopo, u) -> Optional[Iterable]:
    adj = getattr(ftopo, "_vl_adj", None)
    if adj is not None:
        return adj.get(u, [])

    try:
        return ftopo.neighbors(u)
    except Exception:
        pass

    resolver = _ensure_resolver(ftopo)
    fetcher = resolver.get("_fetcher_")
    if u in resolver:
        u_obj = resolver[u]
    elif fetcher:
        try:
            u_obj = fetcher(u)
        except Exception:
            u_obj = None
    else:
        u_obj = None
    if u_obj is not None:
        try:
            return ftopo.neighbors(u_obj)
        except Exception:
            pass

    for fn in ("neighbors_by_id", "neighbours", "neighborsID"):
        if hasattr(ftopo, fn) and callable(getattr(ftopo, fn)):
            try:
                return getattr(ftopo, fn)(u)
            except Exception:
                pass

    try:
        return ftopo.neighbors(str(u))
    except Exception:
        return None

def _neighbors_as_pairs_no_snapshot(ftopo, u: NodeID) -> List[Tuple[NodeID, float]]:
    raw = None
    try:
        raw = ftopo.neighbors(u)
    except Exception:
        pass
    if raw is None:
        resolver = _ensure_resolver(ftopo)
        fetcher = resolver.get("_fetcher_")
        if u in resolver:
            u_obj = resolver[u]
        elif fetcher:
            try:
                u_obj = fetcher(u)
            except Exception:
                u_obj = None
        else:
            u_obj = None
        if u_obj is not None:
            try:
                raw = ftopo.neighbors(u_obj)
            except Exception:
                raw = None
        if raw is None:
            for fn in ("neighbors_by_id", "neighbours", "neighborsID"):
                if hasattr(ftopo, fn) and callable(getattr(ftopo, fn)):
                    try:
                        raw = getattr(ftopo, fn)(u); break
                    except Exception:
                        pass
        if raw is None:
            try:
                raw = ftopo.neighbors(str(u))
            except Exception:
                raw = None

    if not raw or not isinstance(raw, _IterableABC):
        return []

    forward_only = getattr(config, "RING_FORWARD_ONLY", False)

    if isinstance(raw, dict):
        out: List[Tuple[NodeID, float]] = []
        for v, w in raw.items():
            v = getattr(v, "nid", v)
            try:
                w = float(_call_if_callable(w))
            except Exception:
                continue
            if w > MIN_EDGE_MS and (not forward_only or _is_future_neighbor(ftopo, u, v)):
                out.append((v, w))
        return out

    out: List[Tuple[NodeID, float]] = []
    for item in raw:
        if isinstance(item, tuple) and len(item) >= 2:
            v, w = item[0], item[1]
            v = getattr(v, "nid", v)
            try:
                w = float(_call_if_callable(w))
            except Exception:
                continue
            if w > MIN_EDGE_MS and (not forward_only or _is_future_neighbor(ftopo, u, v)):
                out.append((v, w))
            continue
        pair = _edge_to_pair(item)
        if pair is not None:
            v, w = pair
            if w > MIN_EDGE_MS and (not forward_only or _is_future_neighbor(ftopo, u, v)):
                out.append((v, w))
    return out

def _neighbors_as_pairs(ftopo, u: NodeID) -> List[Tuple[NodeID, float]]:
    raw = _neighbors_raw(ftopo, u)
    if not raw or not isinstance(raw, _IterableABC):
        _dbg("No usable neighbors() for node:", u)
        return []

    forward_only = getattr(config, "RING_FORWARD_ONLY", False)

    if isinstance(raw, dict):
        out: List[Tuple[NodeID, float]] = []
        for v, w in raw.items():
            v = getattr(v, "nid", v)
            try:
                w = float(_call_if_callable(w))
            except Exception:
                continue
            if w > MIN_EDGE_MS and (not forward_only or _is_future_neighbor(ftopo, u, v)):
                out.append((v, w))
        return out

    out: List[Tuple[NodeID, float]] = []
    for item in raw:
        if isinstance(item, tuple) and len(item) >= 2:
            v, w = item[0], item[1]
            v = getattr(v, "nid", v)
            try:
                w = float(_call_if_callable(w))
            except Exception:
                continue
            if w > MIN_EDGE_MS and (not forward_only or _is_future_neighbor(ftopo, u, v)):
                out.append((v, w))
            continue
        pair = _edge_to_pair(item)
        if pair is not None:
            v, w = pair
            if w > MIN_EDGE_MS and (not forward_only or _is_future_neighbor(ftopo, u, v)):
                out.append((v, w))
    return out

# --- Optional: one-shot connectivity probe ------------------------------------

def _probe_some_neighbors(ftopo, max_nodes=PROBE_NODES, max_neigh=PROBE_NEIGHBORS):
    try:
        nodes_attr = getattr(ftopo, "nodes", None)
        nodes_iter = nodes_attr() if callable(nodes_attr) else nodes_attr
    except Exception:
        nodes_iter = None
    if not nodes_iter:
        return
    _dbg("=== PROBE: first", max_nodes, "nodes ===")
    count = 0
    for u in nodes_iter:
        if count >= max_nodes:
            break
        u_id = getattr(u, "nid", u)
        raw = _neighbors_raw(ftopo, u_id)
        raw_list = list(raw) if raw and isinstance(raw, _IterableABC) else []
        _dbg("node:", u_id, "raw_neighbors_count:", len(raw_list))
        for it in raw_list[:max_neigh]:
            _dbg("  raw item:", type(it).__name__, repr(it)[:160])
        pairs = _neighbors_as_pairs(ftopo, u_id)
        _dbg("  parsed_pairs:", pairs[:max_neigh])
        count += 1

# --- Algo picker & unified Dijkstra wrapper -----------------------------------

def _get_algo(ftopo) -> Tuple[str, int]:
    cached = getattr(ftopo, "_vl_algo", None)
    if cached is not None:
        return cached
    # Prefer radix if weights are non-negative integer-ish
    if SSSP_ALGO == "binary":
        cached = ("binary", 1)
    elif SSSP_ALGO == "radix":
        ok, scale = _maybe_integerish_weights(ftopo)
        cached = ("radix", (scale if ok and scale > 0 else 1))
    else:  # auto
        ok, scale = _maybe_integerish_weights(ftopo)
        cached = (("radix", scale) if ok else ("binary", 1))
    setattr(ftopo, "_vl_algo", cached)
    return cached

def _dijkstra(
    ftopo,
    src: NodeID,
    dst: NodeID,
    *,
    slo_ms: Optional[float] = None,
) -> Tuple[Optional[float], Dict[NodeID, Optional[NodeID]]]:
    algo, scale = _get_algo(ftopo)
    if algo == "radix":
        return _dijkstra_radix(ftopo, src, dst, slo_ms=slo_ms, scale=scale)
    else:
        return _dijkstra_binary(ftopo, src, dst, slo_ms=slo_ms)

def _slo_key(slo_ms: Optional[float]) -> Optional[int]:
    if slo_ms is None:
        return None
    return getattr(config, "SLO_MS_TASK", None)

# --- Global LRU helpers -------------------------------------------------------

def _global_lru(ftopo) -> OrderedDict:
    if not hasattr(ftopo, "_vl_sssp_lru"):
        setattr(ftopo, "_vl_sssp_lru", OrderedDict())
    return getattr(ftopo, "_vl_sssp_lru")

def _global_key(ftopo, src, sloK):
    view = (
        getattr(config, "RING_FORWARD_ONLY", False),
        getattr(ftopo, "_per_edge_cap", None),  # optional: if your filtered view stores this
        WEIGHT_FROM_RANGE,
    )
    algo = getattr(ftopo, "_vl_algo", ("binary", 1))
    ver  = getattr(ftopo, "version_id", 0)
    return (src, sloK, view, algo, ver)

def _get_global_tree(ftopo, src, sloK, now):
    lru = _global_lru(ftopo)
    key = _global_key(ftopo, src, sloK)
    ent = lru.get(key)
    if ent and (now - ent.tstamp) <= _GLOBAL_SSSP_TTL_SEC:
        lru.move_to_end(key)
        return ent.dist, ent.parent
    return None

def _put_global_tree(ftopo, src, sloK, dist, parent, now):
    lru = _global_lru(ftopo)
    key = _global_key(ftopo, src, sloK)
    lru[key] = _TreeEntry(dist, parent, now)
    lru.move_to_end(key)
    while len(lru) > _GLOBAL_SSSP_LRU_CAP:
        lru.popitem(last=False)

# --- Cached shortest path (adaptive + global LRU, USED BY selection.py) -------

def _pair_shortest_cached(ftopo, src, dst, dijk_cache, slo_ms=None):
    """
    Adaptive strategy:
      • First few distinct dsts for the same (src,slo) -> run pairwise Dijkstra (cheaper).
      • Once fan-out reaches _SSSP_TREE_THRESHOLD -> compute a single tree & reuse.
      • Per-process global LRU of per-source trees to reuse work across stitches.
      • Everything remains backward-compatible with selection.py.
    """
    if SHOW_PROBE:
        flag = getattr(ftopo, "_vl_probe_done", False)
        if not flag:
            setattr(ftopo, "_vl_probe_done", True)
            _probe_some_neighbors(ftopo)

    # Ensure fast adjacency and algo cached
    _ensure_adj_snapshot(ftopo)
    _get_algo(ftopo)

    sloK = _slo_key(slo_ms)
    cache_src = dijk_cache.setdefault(src, {})
    cache_slo = cache_src.setdefault(sloK, {})

    # Quick hit?
    hit = cache_slo.get(dst)
    if hit is not None:
        return hit

    # If a global tree exists, answer from it and plant into per-call cache
    now = time.time()
    gtree = _get_global_tree(ftopo, src, sloK, now)
    if gtree is not None:
        dist_map, parent_map = gtree
        dist = dist_map.get(dst)
        if dist is not None and math.isfinite(dist):
            path = _reconstruct_path(parent_map, src, dst)
            res = (float(dist), path if path and len(path) >= 2 else None)
        else:
            res = (None, None)
        cache_slo[dst] = res
        # Also install the tree into this src bucket so subsequent dsts are fast
        cache_src[("__tree__", sloK)] = (dist_map, parent_map)
        return res

    # Per-(src,slo) counters & tree slots
    cnt_key = ("__cnt__", sloK)
    tree_key = ("__tree__", sloK)

    count = cache_src.get(cnt_key, 0)
    cache_src[cnt_key] = count + 1

    # If we already have a tree (from earlier misses), use it
    tree = cache_src.get(tree_key)
    if tree is not None:
        dist_map, parent_map = tree
        dist = dist_map.get(dst)
        if dist is not None and math.isfinite(dist):
            path = _reconstruct_path(parent_map, src, dst)
            res = (float(dist), path if path and len(path) >= 2 else None)
        else:
            res = (None, None)
        cache_slo[dst] = res
        return res

    # No tree yet: If fan-out still small, do one pairwise Dijkstra
    if count < _SSSP_TREE_THRESHOLD:
        dist, parent = _dijkstra(ftopo, src, dst, slo_ms=slo_ms)
        if dist is None:
            dist, parent = _dijkstra(ftopo, src, dst, slo_ms=None)
        path = _reconstruct_path(parent, src, dst) if dist is not None else None
        res = (dist, path) if (dist and path and len(path) >= 2 and dist > 0.0) else (None, None)
        cache_slo[dst] = res
        return res

    # Threshold reached: compute single-source SSSP tree and serve from it
    dist_map, parent_map = _single_source_sssp(ftopo, src, slo_ms=slo_ms)
    cache_src[tree_key] = (dist_map, parent_map)
    _put_global_tree(ftopo, src, sloK, dist_map, parent_map, now)

    dist = dist_map.get(dst)
    if dist is not None and math.isfinite(dist):
        path = _reconstruct_path(parent_map, src, dst)
        res = (float(dist), path if path and len(path) >= 2 else None)
    else:
        res = (None, None)
    cache_slo[dst] = res
    return res

# --- Heuristics & plumbing ----------------------------------------------------

def _maybe_integerish_weights(ftopo, sample_cap: int = 512, tol: float = 1e-6) -> Tuple[bool, int]:
    _ensure_adj_snapshot(ftopo)
    adj = getattr(ftopo, "_vl_adj", None)

    cnt = 0
    all_int = True
    max_decimals = 0
    sampled: List[float] = []

    if adj:
        for _, vs in adj.items():
            for __, w in vs:
                if w < 0:
                    return (False, 1)
                sampled.append(w)
                frac = abs(w - round(w))
                if frac > tol:
                    all_int = False
                    dec = _decimal_places(w, tol)
                    max_decimals = max(max_decimals, dec)
                cnt += 1
                if cnt >= sample_cap:
                    break
            if cnt >= sample_cap:
                break
    else:
        for _, vs in _iter_some_edges(ftopo, sample_cap):
            for __, w in vs:
                if w < 0:
                    return (False, 1)
                sampled.append(w)
                frac = abs(w - round(w))
                if frac > tol:
                    all_int = False
                    dec = _decimal_places(w, tol)
                    max_decimals = max(max_decimals, dec)
                cnt += 1
                if cnt >= sample_cap:
                    break
            if cnt >= sample_cap:
                break

    if cnt == 0:
        return (False, 1)

    scale = 1 if all_int else (10 ** max_decimals if 0 < max_decimals <= 3 else 1)
    if scale == 1 and not all_int:
        return (False, 1)

    for w in sampled:
        if w > 0:
            iw = int(round(w * scale))
            if iw <= 0:
                return (False, 1)

    return (True, scale)

def _decimal_places(x: float, tol: float) -> int:
    for k in (1, 2, 3):
        y = round(x * (10**k))
        if abs(x * (10**k) - y) < tol * (10**k):
            return k
    return 0

def _iter_some_edges(ftopo, cap: int) -> Iterable[Tuple[NodeID, List[Tuple[NodeID, float]]]]:
    seen = 0
    nodes_iter = None
    if hasattr(ftopo, "nodes"):
        nodes_attr = getattr(ftopo, "nodes")
        nodes_iter = nodes_attr() if callable(nodes_attr) else nodes_attr
    if nodes_iter is None and hasattr(ftopo, "adj"):
        try:
            nodes_iter = getattr(ftopo, "adj").keys()
        except Exception:
            pass
    if nodes_iter is None and hasattr(ftopo, "graph"):
        g = getattr(ftopo, "graph")
        if isinstance(g, dict):
            nodes_iter = g.keys()

    if nodes_iter is None:
        return

    for u in nodes_iter:
        u_id = getattr(u, "nid", u)
        try:
            nbrs = _neighbors_as_pairs(ftopo, u_id)
        except Exception:
            continue
        if not nbrs:
            continue

        yield u_id, nbrs
        seen += len(nbrs)
        if seen >= cap:
            return

def _reconstruct_path(parent: Dict[NodeID, Optional[NodeID]], src: NodeID, dst: NodeID) -> Optional[List[NodeID]]:
    if dst not in parent:
        return None
    path: List[NodeID] = []
    cur: Optional[NodeID] = dst
    while cur is not None:
        path.append(cur)
        if cur == src:
            path.reverse()
            return path
        cur = parent.get(cur)
    return None

# --- Dijkstra (binary heap) ---------------------------------------------------

def _dijkstra_binary(
    ftopo,
    src: NodeID,
    dst: NodeID,
    *,
    slo_ms: Optional[float] = None,
) -> Tuple[Optional[float], Dict[NodeID, Optional[NodeID]]]:
    dist: Dict[NodeID, float] = {src: 0.0}
    parent: Dict[NodeID, Optional[NodeID]] = {src: None}
    pq: List[Tuple[float, NodeID]] = [(0.0, src)]
    best_dst = math.inf

    while pq:
        d, u = heapq.heappop(pq)
        if d != dist.get(u, math.inf):
            continue

        if u == dst:
            best_dst = d
            break

        for v, w in _neighbors_as_pairs(ftopo, u):
            if w <= MIN_EDGE_MS:
                continue
            nd = d + w
            if slo_ms is not None and nd > slo_ms:
                continue
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))

    if best_dst is math.inf and dst in dist:
        best_dst = dist[dst]
    return (None if best_dst is math.inf else float(best_dst), parent)

# --- Dijkstra (radix heap) ----------------------------------------------------

class _RadixHeap:
    def __init__(self):
        self.last = 0
        self.buckets = [[] for _ in range(65)]
        self.size_ = 0

    @staticmethod
    def _msb(x: int) -> int:
        return x.bit_length() - 1

    def _bucket_idx(self, key: int) -> int:
        if key == self.last:
            return 0
        return 1 + self._msb(key ^ self.last)

    def push(self, key: int, val):
        assert key >= self.last
        self.buckets[self._bucket_idx(key)].append((key, val))
        self.size_ += 1

    def _pull(self):
        i = 0
        while i < len(self.buckets) and not self.buckets[i]:
            i += 1
        if i == len(self.buckets):
            return
        new_last = min(k for k, _ in self.buckets[i])
        tmp = self.buckets[i]
        self.buckets[i] = []
        for k, v in tmp:
            idx = 0 if k == new_last else 1 + self._msb(k ^ new_last)
            self.buckets[idx].append((k, v))
        self.last = new_last

    def pop(self):
        if not self.buckets[0]:
            self._pull()
            if not self.buckets[0]:
                return None
        self.size_ -= 1
        return self.buckets[0].pop()

    def __len__(self):
        return self.size_

def _dijkstra_radix(
    ftopo,
    src: NodeID,
    dst: NodeID,
    *,
    slo_ms: Optional[float] = None,
    scale: int = 1,
) -> Tuple[Optional[float], Dict[NodeID, Optional[NodeID]]]:
    INF = (1 << 62)
    dist: Dict[NodeID, int] = {src: 0}
    parent: Dict[NodeID, Optional[NodeID]] = {src: None}
    pq = _RadixHeap()
    pq.push(0, src)

    slo_scaled = None if slo_ms is None else int(math.floor(slo_ms * scale + 1e-9))
    best_dst = INF

    while len(pq):
        cur = pq.pop()
        if cur is None:
            break
        d, u = cur
        if d != dist.get(u, INF):
            continue

        if u == dst:
            best_dst = d
            break

        for v, w in _neighbors_as_pairs(ftopo, u):
            if w <= MIN_EDGE_MS:
                continue
            iw = int(round(w * scale))
            if iw <= 0:
                continue
            nd = d + iw
            if slo_scaled is not None and nd > slo_scaled:
                continue
            if nd < dist.get(v, INF):
                dist[v] = nd
                parent[v] = u
                pq.push(nd, v)

    if best_dst == INF and dst in dist:
        best_dst = dist[dst]
    if best_dst == INF:
        return (None, parent)
    return (best_dst / scale, parent)

# ------------------------------------------------------------------------------
#                           SINGLE-SOURCE SSSP (NEW)
# ------------------------------------------------------------------------------

def _dijkstra_tree_binary(
    ftopo,
    src: NodeID,
    *,
    slo_ms: Optional[float] = None,
) -> Tuple[Dict[NodeID, float], Dict[NodeID, Optional[NodeID]]]:
    dist: Dict[NodeID, float] = {src: 0.0}
    parent: Dict[NodeID, Optional[NodeID]] = {src: None}
    pq: List[Tuple[float, NodeID]] = [(0.0, src)]

    while pq:
        d, u = heapq.heappop(pq)
        if d != dist.get(u, math.inf):
            continue

        for v, w in _neighbors_as_pairs(ftopo, u):
            if w <= MIN_EDGE_MS:
                continue
            nd = d + w
            if slo_ms is not None and nd > slo_ms:
                continue
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))

    return dist, parent

def _dijkstra_tree_radix(
    ftopo,
    src: NodeID,
    *,
    slo_ms: Optional[float] = None,
    scale: int = 1,
) -> Tuple[Dict[NodeID, float], Dict[NodeID, Optional[NodeID]]]:
    INF = (1 << 62)
    dist_i: Dict[NodeID, int] = {src: 0}
    parent: Dict[NodeID, Optional[NodeID]] = {src: None}
    pq = _RadixHeap()
    pq.push(0, src)

    slo_scaled = None if slo_ms is None else int(math.floor(slo_ms * scale + 1e-9))

    while len(pq):
        cur = pq.pop()
        if cur is None:
            break
        d, u = cur
        if d != dist_i.get(u, INF):
            continue

        for v, w in _neighbors_as_pairs(ftopo, u):
            if w <= MIN_EDGE_MS:
                continue
            iw = int(round(w * scale))
            if iw <= 0:
                continue
            nd = d + iw
            if slo_scaled is not None and nd > slo_scaled:
                continue
            if nd < dist_i.get(v, INF):
                dist_i[v] = nd
                parent[v] = u
                pq.push(nd, v)

    dist_f: Dict[NodeID, float] = {u: (di / scale) for u, di in dist_i.items()}
    return dist_f, parent

def _single_source_sssp(
    ftopo,
    src: NodeID,
    *,
    slo_ms: Optional[float] = None,
) -> Tuple[Dict[NodeID, float], Dict[NodeID, Optional[NodeID]]]:
    algo, scale = _get_algo(ftopo)
    if algo == "radix":
        return _dijkstra_tree_radix(ftopo, src, slo_ms=slo_ms, scale=scale)
    else:
        return _dijkstra_tree_binary(ftopo, src, slo_ms=slo_ms)

def _single_source_cached(ftopo, src, tree_cache, slo_ms=None):
    if SHOW_PROBE:
        flag = getattr(ftopo, "_vl_probe_done", False)
        if not flag:
            setattr(ftopo, "_vl_probe_done", True)
            _probe_some_neighbors(ftopo)

    _ensure_adj_snapshot(ftopo)
    _get_algo(ftopo)

    sloK = _slo_key(slo_ms)
    cache_src = tree_cache.setdefault(src, {})
    if sloK in cache_src:
        return cache_src[sloK]
    dist_map, parent_map = _single_source_sssp(ftopo, src, slo_ms=slo_ms)
    cache_src[sloK] = (dist_map, parent_map)
    return cache_src[sloK]

def _path_from_tree(
    parent_map: Dict[NodeID, Optional[NodeID]],
    src: NodeID,
    dst: NodeID
) -> Optional[List[NodeID]]:
    return _reconstruct_path(parent_map, src, dst)
