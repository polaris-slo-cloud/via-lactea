# placement.py

from __future__ import annotations

import functools
import math
import random
from typing import Dict, Iterable, List, Optional, Set, Tuple, Callable

from .profiles import CANDIDATE_STITCHES
from .topology import Node, Topology, dijkstra_base_path_between_nodes


# ---------------------------
# Balanced per-module candidates
# ---------------------------

def assign_modules_to_nodes(
    modules: List[str],
    topo: Topology,
    run_idx: int = 0,
    *,
    allowed_kinds: Optional[Iterable[str]] = None,
    shuffle_once_seed: Optional[int] = None,
    offset_nodes: int = 0,      # kept for API compatibility; unused here
    offset_modules: int = 0,    # kept for API compatibility; unused here
) -> Dict[str, List[Node]]:
    """
    Randomly assign every eligible node exactly one module.
    Returns: module_name -> list[Node] (always lists), shuffled per run.
    Balanced within ±1 across modules. Randomness is reproducible per run.
    """
    if not modules:
        raise ValueError("`modules` must be non-empty.")

    # 1) Eligible nodes
    if allowed_kinds is None:
        eligible_ids = list(topo.nodes.keys())
    else:
        kinds = set(allowed_kinds)
        eligible_ids = [nid for nid, n in topo.nodes.items() if n.kind in kinds]
    if not eligible_ids:
        raise ValueError("No eligible nodes found for placement.")

    # 2) RNG per run (reproducible)
    #seed_val = (0 if shuffle_once_seed is None else int(shuffle_once_seed)) ^ int(run_idx)
    rng = random.Random(random.random())

    # 3) Shuffle nodes
    eligible_ids.sort()
    rng.shuffle(eligible_ids)

    # 4) Balanced module bag
    N, M = len(eligible_ids), len(modules)
    base = N // M
    rem  = N % M

    mods = list(reversed(modules))
    #rng.rever(mods)  # randomize which modules get the +1 first
    counts = {m: base for m in mods}
    for m in mods[:rem]:
        counts[m] += 1

    module_bag: List[str] = []
    for m, c in counts.items():
        module_bag.extend([m] * c)
    rng.shuffle(module_bag)

    # 5) Assign node -> module
    mod_to_nodes: Dict[str, List[Node]] = {m: [] for m in modules}
    for nid, m in zip(eligible_ids, module_bag):
        mod_to_nodes[m].append(topo.nodes[nid])

    # 6) Shuffle each module’s node list (affects greedy iteration order)
    for m in modules:
        rng.shuffle(mod_to_nodes[m])

    return mod_to_nodes


def subsample_module_candidates(
    base: Dict[str, List[Node]],
    rng: random.Random,
    max_per_module: int = 60,
) -> Dict[str, List[Node]]:
    """Per-run, shrink each module's candidate list to at most max_per_module (shuffled)."""
    out: Dict[str, List[Node]] = {}
    for m, lst in base.items():
        if not isinstance(lst, list):
            out[m] = lst
            continue
        pool = lst[:]
        rng.shuffle(pool)
        out[m] = pool[:max_per_module] if max_per_module > 0 else pool
    return out


# ---------------------------
# Edge inventory (computed once)
# ---------------------------

@functools.lru_cache(maxsize=1)
def all_adjacent_module_edges() -> Tuple[Tuple[str, str], ...]:
    edges: Set[Tuple[str, str]] = set()
    for spec in CANDIDATE_STITCHES.values():
        mods = list(reversed(spec["modules"]))
        for i in range(len(mods) - 1):
            edges.add((mods[i], mods[i + 1]))
    return tuple(sorted(edges))


# ---------------------------
# Random per-edge neighbors with reachability + exploration
# ---------------------------

def _bucket_id(n: Node, buckets: int) -> int:
    # Cheap pseudo-cluster without topology attrs; keeps some locality bias
    return hash(n.nid) % max(1, buckets)


def default_get_module_accuracy(module_name: str) -> float:
    """
    Return an accuracy score in [0,1] for a module.
    Replace with your real source (e.g., from CANDIDATE_STITCHES or a table).
    Higher = better accuracy.
    """
    # fallback: unknown modules treated as mid-accuracy
    return 0.5


def build_random_allowed_pairs_for_all_edges(
    topo: Topology,
    base_placement: Dict[str, List[Node]],
    rng: random.Random,
    *,
    k_neighbors: int = 2,
    jitter_neighbors: int = 0,
    # noise / variance knobs
    edge_keep_prob: float = 0.55,   # drop more edges to diversify argmins
    buckets: int = 16,              # pseudo-clusters for "local vs far"
    local_ratio: float = 0.6,       # midpoint for locality if accuracy unknown
    wild_prob: float = 0.12,        # chance to override with 1-2 completely random dsts
    check_reachability: bool = True,
    # accuracy-aware locality controls
    get_module_accuracy: Callable[[str], float] = default_get_module_accuracy,
    local_span: float = 0.30,       # how strongly accuracy swings locality around midpoint
    local_min: float = 0.10,        # clamp lower
    local_max: float = 0.95,        # clamp upper
) -> Dict:
    """
    Fast, variance-friendly per-edge pairs:
      - For each adjacent (m_src -> m_dst), each src picks a mix of local+far dsts.
      - With probability wild_prob, override picks with 1–2 fully random dsts.
      - Optional reachability filter using cached Dijkstra base paths.
      - Drop some pairs with edge_keep_prob to add noise.
      - NEW: Accuracy-aware locality — low-accuracy modules favor more local neighbors.

    Guarantees: at least one pair per edge key if both sides non-empty.
    """
    out: Dict = {}

    # Cache reachability (nid_src, nid_dst) -> bool across ALL edges in this build
    reach_cache: Dict[Tuple[str, str], bool] = {}

    def _reachable(u: Node, v: Node) -> bool:
        if not check_reachability:
            return True
        key = (u.nid, v.nid)
        hit = reach_cache.get(key)
        if hit is not None:
            return hit
        ms, _ = dijkstra_base_path_between_nodes(topo, u.nid, v.nid, float("inf"))
        ok = (ms is not None) and math.isfinite(ms)
        reach_cache[key] = ok
        return ok

    for m_src, m_dst in all_adjacent_module_edges():
        src_nodes = list(base_placement.get(m_src, []))
        dst_nodes = list(base_placement.get(m_dst, []))
        pairs: List[Tuple[Node, Node]] = []

        if not src_nodes or not dst_nodes:
            out[(m_src, m_dst)] = pairs
            continue

        # --- accuracy-aware dynamic local ratio (per source module)
        try:
            acc = float(get_module_accuracy(m_src))  # 0..1
        except Exception:
            acc = 0.5
        # Low accuracy -> push local_ratio up (stay closer). High accuracy -> pull down.
        dyn_local_ratio = local_ratio + local_span * (1.0 - acc)
        if dyn_local_ratio < local_min:
            dyn_local_ratio = local_min
        if dyn_local_ratio > local_max:
            dyn_local_ratio = local_max

        # Precompute dst buckets for quick local/far split
        dst_buckets = [(d, _bucket_id(d, buckets)) for d in dst_nodes]

        for s in src_nodes:
            s_b = _bucket_id(s, buckets)
            local: List[Node] = []
            far: List[Node] = []
            for d, db in dst_buckets:
                (local if db == s_b else far).append(d)

            rng.shuffle(local)
            rng.shuffle(far)

            k_total = max(1, k_neighbors)
            k_local = min(len(local), max(0, int(round(dyn_local_ratio * k_total))))
            k_far   = max(1, k_total - k_local)  # ensure at least one far when possible
            picks: List[Node] = []
            if k_local > 0:
                picks.extend(local[:k_local])
            if k_far > 0:
                picks.extend(far[:min(k_far, len(far))])

            # jitter: optionally add extra random dsts from the remainder
            if jitter_neighbors > 0:
                remainder: List[Node] = []
                if len(local) > k_local:
                    remainder.extend(local[k_local:])
                if len(far) > k_far:
                    remainder.extend(far[k_far:])
                rng.shuffle(remainder)
                picks.extend(remainder[:jitter_neighbors])

            # exploration: sometimes choose 1–2 completely random dsts
            if rng.random() < wild_prob and len(dst_nodes) > 0:
                wild_cnt = 1 if len(dst_nodes) < 3 else 2
                picks = rng.sample(dst_nodes, k=min(wild_cnt, len(dst_nodes)))

            any_added = False
            for d in picks:
                # random edge drop to diversify argmins
                if rng.random() > edge_keep_prob:
                    continue
                if not _reachable(s, d):
                    continue
                pairs.append((s, d))
                any_added = True

            # backstop: keep the edge connected if we dropped everything
            if not any_added:
                pool = dst_nodes[:]
                rng.shuffle(pool)
                chosen = None
                for d in pool:
                    if _reachable(s, d):
                        chosen = d
                        break
                if chosen is None and pool:
                    chosen = pool[0]  # no reachable path: still keep something to avoid empty
                if chosen is not None:
                    pairs.append((s, chosen))

        # Ensure at least one pair if both sides non-empty
        if not pairs:
            s = rng.choice(src_nodes)
            pool = dst_nodes[:]
            rng.shuffle(pool)
            chosen = None
            for d in pool:
                if _reachable(s, d):
                    chosen = d
                    break
            if chosen is None:
                chosen = pool[0]
            pairs.append((s, chosen))

        out[(m_src, m_dst)] = pairs

    return out


def build_local_allowed_pairs_all_stitches(
    topo: Topology,
    base_placement: Dict[str, List[Node]],
    rng: random.Random,
    *,
    # neighborhood size
    k_neighbors: int = 3,
    jitter_neighbors: int = 1,
    # variance knobs
    edge_keep_prob: float = 0.55,
    buckets: int = 16,
    local_ratio: float = 0.6,    # midpoint for locality
    wild_prob: float = 0.12,
    # pool size
    max_per_module: int = 80,
    # safety
    check_reachability: bool = True,
    # accuracy-aware knobs (pass-through)
    get_module_accuracy: Callable[[str], float] = default_get_module_accuracy,
    local_span: float = 0.30,
    local_min: float = 0.10,
    local_max: float = 0.95,
) -> Dict:
    """
    Merge:
      - subsampled module->nodes (per run),
      - per-edge random pairs across all stitches (once per run).
    Keeps output shape compatible with selection layer.

    Accuracy-aware locality is enabled by providing get_module_accuracy (0..1).
    """
    # 1) shrink per-module pools for speed & variance
    shrunk = subsample_module_candidates(base_placement, rng, max_per_module=max_per_module)

    # 2) add fast random neighbors (no per-stitch repetition)
    merged = dict(shrunk)
    merged.update(
        build_random_allowed_pairs_for_all_edges(
            topo=topo,
            base_placement=shrunk,
            rng=rng,
            k_neighbors=k_neighbors,
            jitter_neighbors=jitter_neighbors,
            edge_keep_prob=edge_keep_prob,
            buckets=buckets,
            local_ratio=local_ratio,          # midpoint
            wild_prob=wild_prob,
            check_reachability=check_reachability,
            get_module_accuracy=get_module_accuracy,
            local_span=local_span,
            local_min=local_min,
            local_max=local_max,
        )
    )
    return merged
