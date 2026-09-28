"""Discovery-only rolling-k clustering workers for the semantic sweep."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import modal


RUN_ID = os.environ.get("K_RESOLUTION_RUN_ID", "20260927")
try:
    ROOT = Path(__file__).resolve().parents[2]
except IndexError:  # Modal workers import the module from /root/<script>.py.
    ROOT = Path.cwd()
LOCAL_RUN = ROOT / "semifinal" / "results" / "k_resolution_sweep" / RUN_ID
INPUT = Path("/data/inputs/acquisition-framework-v1-90453ebe")
REMOTE_ROOT = Path("/data/ckpt/k-resolution-sweep-v1")
BACKBONES = ("dinov3", "radio", "aimv2", "siglip2", "dinov2", "siglip", "convnext", "eva02")
PRIMARY4 = ("dinov3", "radio", "siglip2", "convnext")
MB4_MCSF = ("dinov3", "siglip2", "aimv2")
K0 = (4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24)
VOLUME = modal.Volume.from_name("bdc-data", create_if_missing=False)

CORE = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy==1.26.4", "scipy==1.15.3", "pandas==2.3.3", "scikit-learn==1.7.2"
)
CLASSIC_IMAGE = CORE.pip_install("scikit-learn-extra==0.3.0")
MGE_IMAGE = CORE
TORCH_IMAGE = CORE.pip_install("torch==2.5.1")
FGW_IMAGE = CORE.pip_install("POT==0.9.5")
MCSF_IMAGE = (
    modal.Image.from_registry("pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime")
    .pip_install("pandas==2.3.3", "scikit-learn==1.7.2", "scipy==1.15.3", "matplotlib", "PyYAML")
    .run_commands(
        "apt-get update && apt-get install -y git",
        "git clone https://github.com/bingly/MCSF.git /opt/MCSF",
        "git -C /opt/MCSF checkout --detach a19b41d5659ada8cf274be698b3b741e61b699ef",
    )
)

APP_CLASSIC = modal.App("bdc-kroll-classic-20260927")
APP_MGE = modal.App("bdc-kroll-mge-20260927")
APP_MCSF = modal.App("bdc-kroll-mcsf-20260927")
APP_AMF = modal.App("bdc-kroll-amf-20260927")
APP_TURTLE = modal.App("bdc-kroll-turtle-20260927")
APP_FGW = modal.App("bdc-kroll-fgw-20260927")
APP_TEMI = modal.App("bdc-kroll-temi-20260927")
APP_TAC = modal.App("bdc-kroll-tac-20260927")
TEMI_IMAGE = TORCH_IMAGE.add_local_python_source("semantic_g1_core")
TAC_IMAGE = TEMI_IMAGE


def _atomic_json(path: Path, payload: object) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _load_discovery(backbones: tuple[str, ...] = BACKBONES):
    import numpy as np

    keys = np.load(INPUT / "ewaste_keys.npz", allow_pickle=False)["key"].astype(str)
    partitions = np.load(INPUT / "semantic_g0_partitions.npz", allow_pickle=False)
    discovery = partitions["discovery_mask"].astype(bool)
    holdout = partitions["holdout_mask"].astype(bool)
    assert len(keys) == len(np.unique(keys)) == 4179
    assert int(discovery.sum()) == 3579 and int(holdout.sum()) == 600
    assert np.array_equal(partitions["keys"].astype(str), keys)
    assert not np.any(discovery & holdout) and np.all(discovery | holdout)
    embeddings = np.load(INPUT / "ewaste_emb_harm_resjpeg.npz", allow_pickle=False)
    assert np.array_equal(embeddings["key"].astype(str), keys)
    views = {}
    for name in backbones:
        x = np.asarray(embeddings[name], dtype=np.float32)
        assert x.shape[0] == len(keys) and np.isfinite(x).all()
        views[name] = x[discovery].copy()
    return keys[discovery], views


def _load_full_rows():
    import numpy as np
    keys = np.load(INPUT / "ewaste_keys.npz", allow_pickle=False)["key"].astype(str)
    parts = np.load(INPUT / "semantic_g0_partitions.npz", allow_pickle=False)
    discovery = parts["discovery_mask"].astype(bool)
    holdout = parts["holdout_mask"].astype(bool)
    assert len(keys) == len(np.unique(keys)) == 4179
    assert discovery.sum() == 3579 and holdout.sum() == 600 and not np.any(discovery & holdout)
    assert np.all(discovery | holdout) and np.array_equal(parts["keys"].astype(str), keys)
    return keys, discovery, holdout


def _load_npz_embedding(path: Path, key_name: str | None = None):
    import numpy as np
    data = np.load(path, allow_pickle=False)
    keys = data["key"].astype(str)
    if key_name and key_name in data.files:
        values = np.asarray(data[key_name], dtype=np.float32)
    else:
        options = [name for name in data.files if name != "key" and data[name].ndim == 2 and data[name].shape[0] == 4179]
        if len(options) != 1:
            raise ValueError(f"cannot choose unique embedding from {path.name}: {options}")
        values = np.asarray(data[options[0]], dtype=np.float32)
    return keys, values


def _l2(x):
    import numpy as np
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _pca100(x, seed: int = 0):
    from sklearn.decomposition import PCA
    x = _l2(x)
    return _l2(PCA(n_components=min(100, len(x) - 1, x.shape[1]), svd_solver="randomized",
                   whiten=False, random_state=seed).fit_transform(x).astype("float32"))


def _fused_space(views: dict[str, object], names: tuple[str, ...], seed: int = 0):
    from sklearn.decomposition import PCA
    blocks = []
    for name in names:
        x = _l2(views[name])
        blocks.append(PCA(n_components=min(32, len(x) - 1, x.shape[1]), svd_solver="randomized",
                          whiten=False, random_state=seed).fit_transform(x))
    joined = _l2(__import__("numpy").concatenate(blocks, axis=1))
    return _l2(PCA(n_components=min(100, len(joined) - 1, joined.shape[1]), svd_solver="randomized",
                   whiten=False, random_state=seed).fit_transform(joined))


def _save_task(family: str, arm: str, k: int, seed: int, keys, labels, runtime: float, extra=None) -> None:
    import numpy as np
    labels = np.asarray(labels)
    if labels.shape != (3579,) or not np.isfinite(labels).all() or not np.all(labels == np.floor(labels)):
        raise ValueError(f"invalid assignment shape/value: {family}/{arm}/k{k}/seed{seed}")
    labels = labels.astype(np.int32)
    realized = int(np.unique(labels).size)
    if realized != k:
        raise ValueError(f"{family}/{arm} requested k={k}, realized={realized}")
    out = REMOTE_ROOT / family / arm / f"k_{k:02d}" / f"seed_{seed:02d}"
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "labels.npz.tmp"
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, key=keys, labels=labels)
    os.replace(tmp, out / "labels.npz")
    metrics = {"family": family, "arm": arm, "k": k, "seed": seed, "runtime_seconds": runtime,
               "realized_clusters": realized, **(extra or {})}
    _atomic_json(out / "metrics.json", metrics)
    import fcntl
    progress = REMOTE_ROOT / "_progress" / f"{family}.progress.jsonl"
    progress.parent.mkdir(parents=True, exist_ok=True)
    with progress.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps({"family": family, "arm": arm, "k": k, "seed": seed,
                                 "status": "DONE", "runtime_seconds": runtime}, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream, fcntl.LOCK_UN)
    VOLUME.commit()
    print(f"PROGRESS family={family} arm={arm} k={k} seed={seed} status=DONE runtime={runtime:.2f}s", flush=True)


def _task_valid(family: str, arm: str, k: int, seed: int) -> bool:
    import numpy as np
    path = REMOTE_ROOT / family / arm / f"k_{k:02d}" / f"seed_{seed:02d}" / "labels.npz"
    if not path.exists():
        return False
    try:
        data = np.load(path, allow_pickle=False)
        labels = data["labels"]
        return labels.shape == (3579,) and np.isfinite(labels).all() and np.unique(labels).size == k
    except Exception:
        return False


def _chc_tree(distance, linkage: str):
    """CHC hierarchy: constrained NNC components, MST links, then recursive levels."""
    import numpy as np
    base = np.asarray(distance, dtype=np.float32)
    n = len(base)
    if base.shape != (n, n) or n < 2 or not np.isfinite(base).all():
        raise ValueError("CHC requires finite square distance matrix")
    if linkage not in ("single", "average") or np.any(base < 0) or not np.allclose(base, base.T, atol=1e-6):
        raise ValueError("invalid CHC linkage/distance")
    dist = np.full((2 * n, 2 * n), np.inf, dtype=np.float32)
    dist[:n, :n] = base
    np.fill_diagonal(dist, np.inf)
    sizes = np.zeros(2 * n, dtype=np.int32); sizes[:n] = 1
    representative = np.arange(2 * n, dtype=np.int32)
    edges, next_id = [], n
    active_ids = np.arange(n, dtype=np.int32)
    while len(active_ids) > 1:
        current = dist[np.ix_(active_ids, active_ids)]
        np.fill_diagonal(current, np.inf)
        nearest = active_ids[np.argmin(current, axis=1)]
        candidates = {}
        for i0, j0 in zip(active_ids, nearest):
            i, j = int(i0), int(j0)
            if sizes[i] <= sizes[j]:
                pair = (min(i, j), max(i, j))
                d = float(dist[i, j])
                candidates[pair] = (d * d * int(sizes[i]) * int(sizes[j]), pair[0], pair[1])
        if not candidates:
            raise RuntimeError("CHC has no admissible NNC edge")
        parent = np.arange(len(active_ids), dtype=np.int32)
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = int(parent[x])
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb: return False
            parent[rb] = ra; return True
        position = {int(cid): p for p, cid in enumerate(active_ids)}
        for _, a, b in candidates.values(): union(position[a], position[b])
        groups = {}
        for pos in range(len(active_ids)): groups.setdefault(find(pos), []).append(pos)
        components = sorted(groups.values(), key=lambda p: int(active_ids[min(p)]))
        group_index = np.empty(len(active_ids), dtype=np.int32)
        comp_clusters, comp_edges = [], [[] for _ in components]
        for g, positions in enumerate(components):
            group_index[positions] = g; comp_clusters.append(active_ids[positions])
        pos_group = {p: int(group_index[p]) for p in range(len(active_ids))}
        for cost, a, b in candidates.values(): comp_edges[pos_group[position[a]]].append((cost, a, b))
        next_nodes, next_sizes = [], np.empty(len(components), dtype=np.float32)
        for g, clusters in enumerate(comp_clusters):
            if len(clusters) == 1:
                node = int(clusters[0]); next_nodes.append(node); next_sizes[g] = sizes[node]; continue
            local = {int(cid): p for p, cid in enumerate(clusters)}
            tp = np.arange(len(clusters), dtype=np.int32)
            def tf(x):
                while tp[x] != x: tp[x] = tp[tp[x]]; x = int(tp[x])
                return x
            def tu(a, b):
                ra, rb = tf(a), tf(b)
                if ra == rb: return False
                tp[rb] = ra; return True
            selected = 0
            for cost, a, b in sorted(comp_edges[g]):
                if tu(local[a], local[b]):
                    edges.append((int(representative[a]), int(representative[b]), float(cost), len(edges)))
                    selected += 1
                    if selected == len(clusters) - 1: break
            if selected != len(clusters) - 1: raise RuntimeError("CHC component lacks spanning tree")
            node = next_id; next_id += 1
            sizes[node] = int(sizes[clusters].sum()); representative[node] = representative[int(clusters[0])]
            next_nodes.append(node); next_sizes[g] = sizes[node]
        if linkage == "single":
            next_dist = np.full((len(components), len(components)), np.inf, dtype=np.float32)
            np.minimum.at(next_dist, (group_index[:, None], group_index[None, :]), current)
        else:
            weights = sizes[active_ids].astype(np.float32)
            weighted = current * weights[:, None] * weights[None, :]
            numerator = np.zeros((len(components), len(components)), dtype=np.float32)
            np.add.at(numerator, (group_index[:, None], group_index[None, :]), weighted)
            next_dist = numerator / np.maximum(next_sizes[:, None] * next_sizes[None, :], 1.0)
        np.fill_diagonal(next_dist, np.inf)
        active_ids = np.asarray(next_nodes, dtype=np.int32)
        dist[np.ix_(active_ids, active_ids)] = next_dist
        np.fill_diagonal(dist, np.inf)
    if len(edges) != n - 1: raise RuntimeError("incomplete CHC tree")
    return edges


def _cut_chc(edges, n: int, k: int):
    import numpy as np
    parent = np.arange(n, dtype=np.int32)
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = int(parent[x])
        return x
    for a, b, _, _ in sorted(edges, key=lambda e: (e[2], e[3]))[:n-k]:
        ra, rb = find(a), find(b)
        if ra == rb: raise RuntimeError("CHC links do not form a tree")
        parent[rb] = ra
    _, labels = np.unique([find(i) for i in range(n)], return_inverse=True)
    if len(np.unique(labels)) != k: raise RuntimeError("CHC cut returned wrong k")
    return labels.astype(np.int32)


def _classic_one(method: str, arm: str, k: int, seed: int, spaces):
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.mixture import GaussianMixture
    started = time.perf_counter()
    if arm.startswith("kmeans_"):
        rep = arm.removeprefix("kmeans_")
        labels = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(spaces[rep])
    elif arm == "ward_fused":
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(spaces["fused"])
    elif arm == "gmm_diag_fused":
        labels = GaussianMixture(n_components=k, covariance_type="diag", n_init=5, random_state=seed).fit_predict(spaces["fused"])
    elif arm == "kmedoids_pam_fused":
        from sklearn_extra.cluster import KMedoids
        labels = KMedoids(n_clusters=k, metric="euclidean", method="pam", random_state=seed).fit_predict(spaces["fused"])
    elif arm == "evidence_accumulation":
        from scipy.cluster.hierarchy import linkage, cut_tree
        parts = [KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(spaces[name]) for name in BACKBONES]
        coassoc = np.zeros((3579, 3579), dtype=np.float32)
        for part in parts:
            coassoc += part[:, None] == part[None, :]
        distance = 1.0 - coassoc / len(parts)
        labels = cut_tree(linkage(distance[np.triu_indices(3579, 1)], method="average"), n_clusters=k).ravel()
    else:
        raise ValueError(f"unknown classical arm {arm}")
    return np.asarray(labels, dtype=np.int32), time.perf_counter() - started


@APP_CLASSIC.function(image=CLASSIC_IMAGE, cpu=16, memory=32768, timeout=5400, retries=0, volumes={"/data": VOLUME})
def run_classic(tasks: list[dict[str, object]]) -> int:
    import numpy as np
    keys, views = _load_discovery()
    spaces = {name: _pca100(views[name]) for name in BACKBONES}
    spaces["fused"] = _fused_space(views, BACKBONES)
    for task in tasks:
        k, seed, arm = int(task["k"]), int(task["seed"]), str(task["arm"])
        if _task_valid("classic", arm, k, seed):
            print(f"PROGRESS family=classic arm={arm} k={k} seed={seed} status=REUSED", flush=True); continue
        labels, elapsed = _classic_one("classical", arm, k, seed, spaces)
        _save_task("classic", arm, k, seed, keys, labels, elapsed)
    return len(tasks)


@APP_MGE.function(image=MGE_IMAGE, cpu=16, memory=32768, timeout=5400, retries=0, volumes={"/data": VOLUME})
def run_mge_sweep(tasks: list[dict[str, object]]) -> int:
    import numpy as np
    from scipy.spatial.distance import cdist
    keys, views = _load_discovery(("dinov3", "siglip2", "aimv2"))
    distances = []
    for name in ("dinov3", "siglip2", "aimv2"):
        x = _l2(views[name])
        distances.append(cdist(x, x, metric="euclidean").astype(np.float32))
    distance_views = [*distances, np.mean(distances, axis=0, dtype=np.float32)]
    for task in tasks:
        k, seed, linkage = int(task["k"]), int(task["seed"]), str(task["linkage"])
        if seed != 42:
            continue
        arm = f"{linkage}_link"
        if _task_valid("mge", arm, k, seed):
            print(f"PROGRESS family=mge arm={arm} k={k} seed={seed} status=REUSED", flush=True); continue
        started = time.perf_counter()
        count = max(1, int(np.floor(0.5 * k + 0.5)))
        granularities = sorted(set(np.floor(np.linspace(k, 1, count) + 0.5).astype(int).tolist()), reverse=True)
        partitions, memberships = [], []
        for d in distance_views:
            tree = _chc_tree(d, linkage)
            view_labels = [_cut_chc(tree, 3579, q) for q in granularities]
            partitions.append(view_labels)
            for labels in view_labels:
                membership = np.zeros((3579, int(labels.max()) + 1), dtype=np.uint8)
                membership[np.arange(3579), labels] = 1
                memberships.append(membership)
        membership = np.concatenate(memberships, axis=1).astype(np.int32)
        sizes = membership.sum(axis=0, dtype=np.int32).astype(np.float32)
        intersections = membership.T @ membership
        jaccard = intersections.astype(np.float32) / np.maximum(sizes[:, None] + sizes[None, :] - intersections, 1)
        np.fill_diagonal(jaccard, 0)
        row_sums = jaccard.sum(axis=1, keepdims=True)
        isolated = row_sums[:, 0] <= 0
        if isolated.any():
            jaccard[isolated, isolated] = 1.0
            row_sums[isolated] = 1.0
        transition = jaccard / row_sums
        powers, power = [], transition.copy()
        for _ in range(20):
            powers.append(power)
            power = power @ transition
        trajectory = np.concatenate(powers, axis=1)
        trajectory /= np.maximum(np.linalg.norm(trajectory, axis=1, keepdims=True), 1e-12)
        node_sim = np.clip(trajectory @ trajectory.T, 0, 1).astype(np.float32)
        coassociation = np.zeros((3579, 3579), dtype=np.float32)
        offset = 0
        weight = 1.0 / len(memberships)
        for view_labels in partitions:
            for labels in view_labels:
                q = int(labels.max()) + 1
                coassociation += node_sim[offset:offset + q, offset:offset + q][np.ix_(labels, labels)] * weight
                offset += q
        final_tree = _chc_tree(np.clip(1 - coassociation, 0, 1), linkage)
        labels = _cut_chc(final_tree, 3579, k)
        _save_task("mge", arm, k, seed, keys, labels, time.perf_counter() - started,
                   {"linkage": linkage, "lambda": 0.5, "t": 20, "granularity_counts": granularities})
    return len(tasks)


@APP_MCSF.function(image=MCSF_IMAGE, gpu="A100-40GB", cpu=4, memory=16384, timeout=5400, retries=0, volumes={"/data": VOLUME})
def run_mcsf_sweep(tasks: list[dict[str, object]]) -> int:
    import sys
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.cluster import KMeans
    from torch.utils.data import DataLoader, Dataset
    if not torch.cuda.is_available():
        raise RuntimeError("MCSF parity requires CUDA")
    sys.path.insert(0, "/opt/MCSF/MCSF")
    from layers import MCSFNetwork
    import utils as mcsf_utils
    keys, views = _load_discovery(MB4_MCSF)
    device = torch.device("cuda")
    arrays = [torch.from_numpy(_l2(views[name])).to(device) for name in MB4_MCSF]
    class PrecomputedViews(Dataset):
        def __len__(self): return len(arrays[0])
        def __getitem__(self, index): return [view[index] for view in arrays], index
    for task in tasks:
        k, seed = int(task["k"]), int(task["seed"])
        if _task_valid("mcsf", "mb3", k, seed):
            print(f"PROGRESS family=mcsf arm=mb3 k={k} seed={seed} status=REUSED", flush=True); continue
        started = time.perf_counter()
        mcsf_utils.set_seed(seed)
        loader = DataLoader(PrecomputedViews(), batch_size=3579, shuffle=False, drop_last=False)
        batch_views, row_ids = next(iter(loader))
        assert torch.equal(row_ids, torch.arange(3579))
        batch_views = [x.to(device) for x in batch_views]
        positive = mcsf_utils.adj_graphs(batch_views, 3579, 15, "cosine")
        adjacency = torch.as_tensor(mcsf_utils.fused_adj_graph(positive, 3579, 3), dtype=torch.float32, device=device)
        model = MCSFNetwork(3, np.asarray([x.shape[1] for x in arrays]), [256, 512, 1024], 1500, 1024, k, 3579).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0)
        for epoch in range(201):
            model.train()
            probs, recon, consensus, sims = model(batch_views)
            loss = model.loss(batch_views, probs, recon, sims, adjacency, 1.0, 1.0, 0.01)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"MCSF nonfinite loss: k={k} seed={seed} epoch={epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if epoch in (0, 100, 200):
                print(f"PROGRESS family=mcsf arm=mb3 k={k} seed={seed} epoch={epoch} device={torch.cuda.get_device_name(0)} loss={loss.item():.6g}", flush=True)
        model.eval()
        with torch.no_grad():
            _, _, consensus, _ = model(batch_views)
        consensus = consensus.detach().cpu().numpy().astype(np.float32)
        assert consensus.shape == (3579, k)
        labels = KMeans(n_clusters=k, n_init=20, random_state=23).fit_predict(consensus)
        _save_task("mcsf", "mb3", k, seed, keys, labels, time.perf_counter() - started,
                   {"epochs": 201, "device": torch.cuda.get_device_name(0)})
    return len(tasks)


def _amf_spectral(views, k: int, seed: int, adaptive: bool):
    import numpy as np
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import eigsh
    from sklearn.cluster import KMeans
    distances = [np.maximum((x * x).sum(1, keepdims=True) + (x * x).sum(1)[None, :] - 2 * (x @ x.T), 0).astype(np.float32) for x in views]
    alpha = np.full(len(views), 1.0 / len(views), dtype=np.float64)
    spectral = None
    history = []
    for iteration in range(50):
        fused = sum(d / max(1.0 - alpha[i], 1e-6) for i, d in enumerate(distances))
        if spectral is not None:
            fused += 0.1 * np.maximum((spectral * spectral).sum(1, keepdims=True) + (spectral * spectral).sum(1)[None, :] - 2 * (spectral @ spectral.T), 0)
        work = fused.copy(); np.fill_diagonal(work, np.inf)
        knn = 30; idx = np.argpartition(work, kth=knn, axis=1)[:, :knn + 1]
        vals = np.take_along_axis(work, idx, axis=1); order = np.argsort(vals, axis=1, kind="stable")
        idx, vals = np.take_along_axis(idx, order, axis=1), np.take_along_axis(vals, order, axis=1)
        edge, nearest = vals[:, knn], vals[:, :knn]
        denom = np.maximum(knn * edge - nearest.sum(1), 1e-12)
        weights = np.maximum((edge[:, None] - nearest) / denom[:, None], 0)
        graph = np.zeros_like(work, dtype=np.float32); graph[np.arange(len(work))[:, None], idx[:, :knn]] = weights
        symmetric = 0.5 * (graph + graph.T)
        lap = np.diag(symmetric.sum(1)) - symmetric
        phi = np.maximum(np.asarray([(symmetric * d).sum() for d in distances]), 1e-10)
        if adaptive:
            # Correct adaptive update: alpha is solved from per-view graph costs.
            order_alpha = np.argsort(phi); root = np.sqrt(phi[order_alpha]); cumulative = np.cumsum(root)
            active = len(phi)
            for candidate in range(2, len(phi) + 1):
                total = cumulative[candidate - 1]
                lower = total / root[candidate] + 1 if candidate < len(phi) else -np.inf
                upper = total / root[candidate - 1] + 1
                if lower <= candidate < upper:
                    active = candidate; break
            total = cumulative[active - 1]; alpha = np.zeros(len(phi), dtype=np.float64)
            alpha[order_alpha[:active]] = np.maximum(0, 1 - (active - 1) * root[:active] / total)
        vals_eig, spectral = eigsh(sparse.csr_matrix(lap), k=k, which="SM", tol=1e-3, maxiter=3000,
                                  v0=np.random.default_rng(seed + iteration).normal(size=len(lap)))
        order = np.argsort(vals_eig); vals_eig, spectral = vals_eig[order], spectral[:, order]
        objective = float(np.sum(phi / np.maximum(1 - alpha, 1e-6)) + np.sum(denom * (weights * weights).sum(1)) + 0.2 * np.sum(vals_eig))
        history.append(objective)
        if iteration and abs(history[-2] - objective) / max(abs(history[-2]), 1e-8) < 1e-3: break
    labels = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(spectral).astype(np.int32)
    return labels, alpha, len(history)


@APP_AMF.function(image=CORE, cpu=16, memory=32768, timeout=5400, retries=0, volumes={"/data": VOLUME})
def run_amf_sweep(tasks: list[dict[str, object]]) -> int:
    keys, views = _load_discovery()
    spaces = {name: _pca100(views[name]) for name in BACKBONES}
    for task in tasks:
        arm, k, seed = str(task["arm"]), int(task["k"]), int(task["seed"])
        if _task_valid("amf", arm, k, seed):
            print(f"PROGRESS family=amf arm={arm} k={k} seed={seed} status=REUSED", flush=True); continue
        names = PRIMARY4 if arm.startswith("mb4") else BACKBONES
        adaptive = arm.endswith("adaptive")
        started = time.perf_counter()
        labels, alpha, iterations = _amf_spectral([spaces[name] for name in names], k, seed, adaptive)
        _save_task("amf", arm, k, seed, keys, labels, time.perf_counter() - started,
                   {"alpha": alpha.tolist(), "objective_iterations": iterations, "graph_knn": 30})
    return len(tasks)


def _turtle_train(views, k: int, seed: int, steps: int, device: str):
    import random
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.set_num_threads(min(8, os.cpu_count() or 1)); torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    target = torch.device(device)
    tensors = [torch.from_numpy(_l2(x)).to(target) for x in views]
    encoders = [nn.utils.weight_norm(nn.Linear(x.shape[1], k)).to(target) for x in tensors]
    optimizer = torch.optim.Adam([p for module in encoders for p in module.parameters()], lr=0.001, betas=(0.9, 0.999))
    def encode(batch):
        per_view = [F.softmax(module(x), dim=1) for module, x in zip(encoders, batch)]
        return torch.mean(torch.stack(per_view), dim=0), per_view
    trace = []
    for iteration in range(steps):
        ids = np.random.choice(len(tensors[0]), size=len(tensors[0]), replace=False)
        batch = [x[ids] for x in tensors]; labels, per_view = encode(batch)
        inner = [nn.Linear(x.shape[1], k).to(target) for x in tensors]
        inner_opt = torch.optim.Adam([p for m in inner for p in m.parameters()], lr=0.001, betas=(0.9, 0.999))
        for _ in range(10):
            inner_opt.zero_grad(); loss = sum(F.cross_entropy(m(x), labels.detach()) for m, x in zip(inner, batch)); loss.backward(); inner_opt.step()
        optimizer.zero_grad(); pred = sum(F.cross_entropy(m(x).detach(), labels) for m, x in zip(inner, batch))
        entropy = sum(torch.special.entr(prob.mean(0)).sum() for prob in per_view)
        (pred - 10 * entropy).backward(); optimizer.step()
        trace.append(float((pred - 10 * entropy).detach()))
        if steps >= 6000 and (iteration + 1) % 1000 == 0:
            print(f"PROGRESS family=turtle k={k} seed={seed} step={iteration+1}/{steps} device={target}", flush=True)
    with torch.inference_mode():
        scores, _ = encode(tensors)
    labels = scores.argmax(1).cpu().numpy().astype(np.int32)
    return labels, trace


@APP_TURTLE.function(image=TORCH_IMAGE, gpu="A10", cpu=8, memory=24576, timeout=5400, retries=0, volumes={"/data": VOLUME})
def benchmark_turtle() -> dict[str, object]:
    import json
    keys, views = _load_discovery()
    selected = [views[name] for name in PRIMARY4]
    records = {}
    for device in ("cpu", "cuda"):
        t0 = time.perf_counter(); labels, _ = _turtle_train(selected, 12, 42, 200, device)
        records[device] = {"seconds": time.perf_counter() - t0, "clusters": int(len(set(labels.tolist())))}
    records["gpu_speedup"] = records["cpu"]["seconds"] / max(records["cuda"]["seconds"], 1e-9)
    records["decision"] = "A10" if records["gpu_speedup"] >= 1.5 and records["cuda"]["clusters"] == 12 else "CPU16"
    path = REMOTE_ROOT / "_progress" / "turtle_benchmark.json"; path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(path, records); VOLUME.commit()
    print("PROGRESS family=turtle benchmark=" + json.dumps(records), flush=True)
    return records


@APP_TURTLE.function(image=TORCH_IMAGE, gpu="A10", cpu=8, memory=24576, timeout=5400, retries=0, volumes={"/data": VOLUME})
def run_turtle_gpu(tasks: list[dict[str, object]]) -> int:
    keys, views = _load_discovery()
    for task in tasks:
        arm, k, seed = str(task["arm"]), int(task["k"]), int(task["seed"])
        if _task_valid("turtle", arm, k, seed):
            print(f"PROGRESS family=turtle arm={arm} k={k} seed={seed} status=REUSED", flush=True); continue
        names = PRIMARY4 if arm == "mb4" else BACKBONES
        started = time.perf_counter(); labels, _ = _turtle_train([views[n] for n in names], k, seed, 6000, "cuda")
        _save_task("turtle", arm, k, seed, keys, labels, time.perf_counter()-started, {"outer_steps": 6000, "device": "A10"})
    return len(tasks)


@APP_TURTLE.function(image=TORCH_IMAGE, cpu=16, memory=32768, timeout=5400, retries=0, volumes={"/data": VOLUME})
def run_turtle_cpu(tasks: list[dict[str, object]]) -> int:
    keys, views = _load_discovery()
    for task in tasks:
        arm, k, seed = str(task["arm"]), int(task["k"]), int(task["seed"])
        if _task_valid("fgw", arm, k, seed):
            print(f"PROGRESS family=fgw arm={arm} k={k} seed={seed} status=REUSED", flush=True); continue
        names = PRIMARY4 if arm == "mb4" else BACKBONES
        started = time.perf_counter(); labels, _ = _turtle_train([views[n] for n in names], k, seed, 6000, "cpu")
        _save_task("turtle", arm, k, seed, keys, labels, time.perf_counter()-started, {"outer_steps": 6000, "device": "CPU16"})
    return len(tasks)


@APP_FGW.function(image=FGW_IMAGE, cpu=16, memory=32768, timeout=5400, retries=0, volumes={"/data": VOLUME})
def run_fgw_sweep(tasks: list[dict[str, object]]) -> int:
    import numpy as np
    import ot
    from scipy.optimize import linear_sum_assignment
    from sklearn.cluster import KMeans
    keys, views = _load_discovery()
    stats = __import__("pandas").read_csv(INPUT / "imgstats.csv").set_index("key").loc[np.load(INPUT / "ewaste_keys.npz", allow_pickle=False)["key"].astype(str)]
    source_all = (np.maximum(stats.W.to_numpy(), stats.H.to_numpy()) > 150).astype(np.int8)
    discovery = np.load(INPUT / "semantic_g0_partitions.npz", allow_pickle=False)["discovery_mask"].astype(bool)
    source = source_all[discovery]
    ids0, ids1 = np.flatnonzero(source == 0), np.flatnonzero(source == 1)
    for task in tasks:
        arm, k, seed = str(task["arm"]), int(task["k"]), int(task["seed"])
        names = PRIMARY4 if arm == "mb4" else BACKBONES
        started = time.perf_counter(); fused = _fused_space(views, names)
        a = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(fused[ids0])
        b = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(fused[ids1])
        def cos(x, y=None):
            x = _l2(x); y = x if y is None else _l2(y); return np.clip(1 - x @ y.T, 0, 2).astype(np.float64)
        cross = cos(a.cluster_centers_, b.cluster_centers_)
        cost = ot.gromov.fused_gromov_wasserstein(cross, cos(a.cluster_centers_), cos(b.cluster_centers_),
                 np.full(k, 1/k), np.full(k, 1/k), loss_fun="square_loss", alpha=0.5, log=False, verbose=False)
        row, col = linear_sum_assignment(-cost); mapping = np.empty(k, dtype=np.int32); mapping[col] = row
        labels = np.empty(len(source), dtype=np.int32); labels[ids0] = a.labels_; labels[ids1] = mapping[b.labels_]
        _save_task("fgw", arm, k, seed, keys, labels, time.perf_counter()-started, {"alpha": 0.5})
    return len(tasks)


@APP_TEMI.function(image=TEMI_IMAGE, gpu="A10", cpu=4, memory=16384, timeout=5400, retries=0, volumes={"/data": VOLUME})
def run_temi_sweep(tasks: list[dict[str, object]]) -> int:
    import numpy as np
    import torch
    from semantic_g1_core import train_temi
    if not torch.cuda.is_available():
        raise RuntimeError("TEMI requires CUDA")
    keys, fit_mask, _ = _load_full_rows()
    names = ("dinov2", "radio", "siglip2")
    full = np.load(INPUT / "ewaste_emb_harm_resjpeg.npz", allow_pickle=False)
    radio = np.load(Path("/data/ckpt/semantic-g1/radio_adapters.npz"), allow_pickle=False)
    radio_keys = radio["key"].astype(str)
    assert np.array_equal(radio_keys, keys)
    g2_root = Path("/data/ckpt/semantic-g2")
    reps = {name: np.asarray(full[name], dtype=np.float32) for name in names if name != "radio"}
    reps["radio"] = np.asarray(full["radio"], dtype=np.float32)
    for name, filename in (("msn", "embedding_msn.npz"), ("franca", "embedding_franca.npz"),
                           ("fashionsiglip", "embedding_fashionsiglip.npz")):
        path = g2_root / filename
        if not path.exists():
            matches = sorted(g2_root.glob(f"*{name}*.npz"))
            if not matches:
                print(f"PROGRESS family=temi arm={name} status=MISSING", flush=True); continue
            path = matches[0]
        cache_keys, values = _load_npz_embedding(path)
        if not np.array_equal(cache_keys, keys): raise ValueError(f"key order mismatch: {path}")
        reps[name] = values
    for task in tasks:
        representation, k, seed = str(task["representation"]), int(task["k"]), int(task.get("seed", 42))
        if representation not in reps:
            continue
        if _task_valid("temi", representation, k, seed):
            print(f"PROGRESS family=temi arm={representation} k={k} seed={seed} status=REUSED", flush=True); continue
        started = time.perf_counter()
        result = train_temi(reps[representation], fit_mask, seed=seed, epochs=100, n_heads=16,
                            n_clusters=k, knn=50, batch_size=512)
        labels = np.asarray(result["labels"][int(result["best_head"])])[fit_mask]
        _save_task("temi", representation, k, seed, keys[fit_mask], labels,
                   time.perf_counter()-started, {"epochs": 100, "heads": 16, "best_head": int(result["best_head"])})
    return len(tasks)


@APP_TAC.function(image=TAC_IMAGE, gpu="A10", cpu=4, memory=16384, timeout=5400, retries=0, volumes={"/data": VOLUME})
def run_tac_sweep(tasks: list[dict[str, object]]) -> int:
    import numpy as np
    import torch
    from scipy.special import softmax
    from semantic_g1_core import choose_discriminative_nouns, pca_all, spherical_kmeans_all, train_tac_heads, l2
    if not torch.cuda.is_available():
        raise RuntimeError("TAC requires CUDA")
    keys, fit_mask, _ = _load_full_rows()
    core = np.load(INPUT / "ewaste_emb_harm_resjpeg.npz", allow_pickle=False)
    radio = np.load(Path("/data/ckpt/semantic-g1/radio_adapters.npz"), allow_pickle=False)
    assert np.array_equal(radio["key"].astype(str), keys)
    wordnet_root = Path("/data/ckpt/semantic-g1")
    for task in tasks:
        representation, k = str(task["representation"]), int(task["k"])
        if representation == "siglip2":
            image = np.asarray(core["siglip2"], dtype=np.float32); noun_path = wordnet_root / "wordnet_siglip2.npz"
        elif representation == "radio_siglip2g":
            image = np.asarray(radio["siglip2-g"], dtype=np.float32); noun_path = wordnet_root / "wordnet_radio_siglip2g.npz"
        else:
            raise ValueError(f"unknown TAC representation {representation}")
        data = np.load(noun_path, allow_pickle=False)
        nouns = np.asarray(data["nouns"]).astype(str)
        noun_embeddings = l2(np.asarray(data["embeddings"], dtype=np.float32))
        assert noun_embeddings.shape[0] == len(nouns) and image.shape[0] == len(keys)
        arms = (f"{representation}_initial", f"{representation}_concat", f"{representation}_head")
        if all(_task_valid("tac", arm, k, seed) for arm, seed in ((arms[0], 42), (arms[1], 42),
                                                                  (arms[2], 42), (arms[2], 43), (arms[2], 44))):
            print(f"PROGRESS family=tac arm={representation} k={k} status=REUSED", flush=True); continue
        started = time.perf_counter()
        centers_labels, centers = spherical_kmeans_all(image, fit_mask, n_clusters=k, seed=0)
        selected, _, _ = choose_discriminative_nouns(centers, noun_embeddings, top_k=5)
        selected_embeddings = noun_embeddings[selected]
        text = np.empty_like(image, dtype=np.float32)
        for start in range(0, len(image), 512):
            weights = softmax(l2(image[start:start+512]) @ selected_embeddings.T / 0.005, axis=1)
            text[start:start+len(weights)] = l2(weights @ selected_embeddings)
        concatenated = np.concatenate((l2(image), text), axis=1)
        concat_pca = pca_all(concatenated, fit_mask, 100, 0)[0]
        concat_labels, _ = spherical_kmeans_all(concat_pca, fit_mask, n_clusters=k, seed=42)
        head = train_tac_heads(image, text, fit_mask, seeds=(42, 43, 44), epochs=20,
                               n_clusters=k, knn=50, batch_size=512)
        _save_task("tac", arms[0], k, 42, keys[fit_mask], centers_labels[fit_mask],
                   time.perf_counter()-started, {"source": "initial_image_spherical_kmeans"})
        _save_task("tac", arms[1], k, 42, keys[fit_mask], concat_labels[fit_mask],
                   time.perf_counter()-started, {"source": "concat_spherical_kmeans"})
        for index, seed in enumerate((42, 43, 44)):
            _save_task("tac", arms[2], k, seed, keys[fit_mask], head["labels"][index][fit_mask],
                       time.perf_counter()-started, {"epochs": 20, "selected_nouns": len(selected)})
    return len(tasks)


@APP_CLASSIC.local_entrypoint()
def launch_parity() -> None:
    LOCAL_RUN.mkdir(parents=True, exist_ok=True)
    classic_tasks = [{"arm": f"kmeans_{name}", "k": 12, "seed": seed}
                     for name in (*BACKBONES, "fused") for seed in (42, 43, 44)]
    classic_tasks += [{"arm": arm, "k": 12, "seed": seed}
                      for arm in ("ward_fused", "gmm_diag_fused", "kmedoids_pam_fused", "evidence_accumulation")
                      for seed in ((42, 43, 44) if arm != "ward_fused" else (42,))]
    call = run_classic.spawn(classic_tasks)
    _record_call("classic", "bdc-kroll-classic-20260927", call, "parity_k12")


def _record_call(family: str, app: str, call, phase: str) -> None:
    LOCAL_RUN.mkdir(parents=True, exist_ok=True)
    path = LOCAL_RUN / f"launcher_{family}.json"
    _atomic_json(path, {"run_id": RUN_ID, "started_utc": datetime.now(timezone.utc).isoformat(),
                        "app": app, "function_call_id": call.object_id, "phase": phase})
    print(f"APP={app} family={family} function_call={call.object_id} phase={phase}", flush=True)


@APP_MGE.local_entrypoint()
def launch_mge_parity() -> None:
    tasks = [{"linkage": linkage, "k": 12, "seed": 42} for linkage in ("single", "average")]
    _record_call("mge", "bdc-kroll-mge-20260927", run_mge_sweep.spawn(tasks), "parity_k12")


@APP_MCSF.local_entrypoint()
def launch_mcsf_parity() -> None:
    for seed in (42, 43, 44):
        call = run_mcsf_sweep.spawn([{"k": 12, "seed": seed}])
        _record_call(f"mcsf_s{seed}", "bdc-kroll-mcsf-20260927", call, "parity_k12")


def _shards(tasks: list[dict[str, object]], count: int = 4) -> list[list[dict[str, object]]]:
    return [tasks[i::count] for i in range(count) if tasks[i::count]]


def _launch_shard(family: str, app_name: str, runner, tasks: list[dict[str, object]], shard: int) -> None:
    shards = _shards(tasks)
    if shard < 0 or shard >= len(shards): raise ValueError(f"shard index must be 0..{len(shards)-1}")
    _record_call(f"{family}_shard{shard}", app_name, runner.spawn(shards[shard]), "coarse_k")


@APP_CLASSIC.local_entrypoint()
def launch_classic_sweep() -> None:
    arms = [*(f"kmeans_{name}" for name in (*BACKBONES, "fused")), "ward_fused",
            "gmm_diag_fused", "kmedoids_pam_fused", "evidence_accumulation"]
    tasks = [{"arm": arm, "k": k, "seed": seed}
             for k in K0 for arm in arms for seed in ((42,) if arm == "ward_fused" else (42, 43, 44))]
    for index, shard in enumerate(_shards(tasks)):
        _record_call(f"classic_shard{index}", "bdc-kroll-classic-20260927", run_classic.spawn(shard), "coarse_k")


@APP_MGE.local_entrypoint()
def launch_mge_sweep() -> None:
    tasks = [{"linkage": linkage, "k": k, "seed": 42} for k in K0 for linkage in ("single", "average")]
    for index, shard in enumerate(_shards(tasks)):
        _record_call(f"mge_shard{index}", "bdc-kroll-mge-20260927", run_mge_sweep.spawn(shard), "coarse_k")


@APP_MCSF.local_entrypoint()
def launch_mcsf_sweep() -> None:
    tasks = [{"k": k, "seed": seed} for k in K0 for seed in (42, 43, 44)]
    for index, shard in enumerate(_shards(tasks)):
        _record_call(f"mcsf_shard{index}", "bdc-kroll-mcsf-20260927", run_mcsf_sweep.spawn(shard), "coarse_k")


@APP_AMF.local_entrypoint()
def launch_amf_parity() -> None:
    tasks = [{"arm": f"{views}_{mode}", "k": 12, "seed": seed}
             for views in ("mb4", "mb8") for mode in ("uniform", "adaptive") for seed in (42, 43, 44)]
    _record_call("amf", "bdc-kroll-amf-20260927", run_amf_sweep.spawn(tasks), "parity_k12")


@APP_AMF.local_entrypoint()
def launch_amf_sweep() -> None:
    tasks = [{"arm": f"{views}_{mode}", "k": k, "seed": seed}
             for k in K0 for views in ("mb4", "mb8") for mode in ("uniform", "adaptive") for seed in (42, 43, 44)]
    for index, shard in enumerate(_shards(tasks)):
        _record_call(f"amf_shard{index}", "bdc-kroll-amf-20260927", run_amf_sweep.spawn(shard), "coarse_k")


@APP_FGW.local_entrypoint()
def launch_fgw_parity() -> None:
    tasks = [{"arm": arm, "k": 12, "seed": 42} for arm in ("mb4", "mb8")]
    _record_call("fgw", "bdc-kroll-fgw-20260927", run_fgw_sweep.spawn(tasks), "parity_k12")


@APP_FGW.local_entrypoint()
def launch_fgw_sweep() -> None:
    tasks = [{"arm": arm, "k": k, "seed": 42} for k in K0 for arm in ("mb4", "mb8")]
    for index, shard in enumerate(_shards(tasks)):
        _record_call(f"fgw_shard{index}", "bdc-kroll-fgw-20260927", run_fgw_sweep.spawn(shard), "coarse_k")


@APP_TURTLE.local_entrypoint()
def launch_turtle_benchmark() -> None:
    _record_call("turtle_benchmark", "bdc-kroll-turtle-20260927", benchmark_turtle.spawn(), "cpu_vs_a10_200_steps")


@APP_TURTLE.local_entrypoint()
def launch_turtle_sweep() -> None:
    tasks = [{"arm": arm, "k": k, "seed": 42} for k in K0 for arm in ("mb4", "mb8")]
    tasks += [{"arm": arm, "k": 12, "seed": seed} for arm in ("mb4", "mb8") for seed in (43, 44)]
    for index, shard in enumerate(_shards(tasks)):
        _record_call(f"turtle_shard{index}", "bdc-kroll-turtle-20260927", run_turtle_gpu.spawn(shard), "coarse_k")


@APP_TEMI.local_entrypoint()
def launch_temi_parity() -> None:
    tasks = [{"representation": name, "k": 12, "seed": 42}
             for name in ("dinov2", "radio", "siglip2", "msn", "franca", "fashionsiglip")]
    for index, shard in enumerate(_shards(tasks)):
        _record_call(f"temi_parity{index}", "bdc-kroll-temi-20260927", run_temi_sweep.spawn(shard), "parity_k12")


@APP_TEMI.local_entrypoint()
def launch_temi_sweep() -> None:
    tasks = [{"representation": name, "k": k, "seed": 42}
             for k in K0 for name in ("dinov2", "radio", "siglip2", "msn", "franca", "fashionsiglip")]
    for index, shard in enumerate(_shards(tasks)):
        _record_call(f"temi_shard{index}", "bdc-kroll-temi-20260927", run_temi_sweep.spawn(shard), "coarse_k")


@APP_TAC.local_entrypoint()
def launch_tac_parity() -> None:
    tasks = [{"representation": name, "k": 12} for name in ("siglip2", "radio_siglip2g")]
    _record_call("tac_parity", "bdc-kroll-tac-20260927", run_tac_sweep.spawn(tasks), "parity_k12")


@APP_TAC.local_entrypoint()
def launch_tac_sweep() -> None:
    tasks = [{"representation": name, "k": k}
             for k in K0 for name in ("siglip2", "radio_siglip2g")]
    for index, shard in enumerate(_shards(tasks)):
        _record_call(f"tac_shard{index}", "bdc-kroll-tac-20260927", run_tac_sweep.spawn(shard), "coarse_k")


@APP_CLASSIC.local_entrypoint()
def launch_classic_shard(shard: int = 0) -> None:
    arms = [*(f"kmeans_{name}" for name in (*BACKBONES, "fused")), "ward_fused",
            "gmm_diag_fused", "kmedoids_pam_fused", "evidence_accumulation"]
    tasks = [{"arm": arm, "k": k, "seed": seed}
             for k in K0 for arm in arms for seed in ((42,) if arm == "ward_fused" else (42, 43, 44))]
    _launch_shard("classic", "bdc-kroll-classic-20260927", run_classic, tasks, shard)


@APP_MGE.local_entrypoint()
def launch_mge_shard(shard: int = 0) -> None:
    tasks = [{"linkage": linkage, "k": k, "seed": 42} for k in K0 for linkage in ("single", "average")]
    _launch_shard("mge", "bdc-kroll-mge-20260927", run_mge_sweep, tasks, shard)


@APP_MCSF.local_entrypoint()
def launch_mcsf_shard(shard: int = 0) -> None:
    tasks = [{"k": k, "seed": seed} for k in K0 for seed in (42, 43, 44)]
    _launch_shard("mcsf", "bdc-kroll-mcsf-20260927", run_mcsf_sweep, tasks, shard)


@APP_AMF.local_entrypoint()
def launch_amf_shard(shard: int = 0) -> None:
    tasks = [{"arm": f"{views}_{mode}", "k": k, "seed": seed}
             for k in K0 for views in ("mb4", "mb8") for mode in ("uniform", "adaptive") for seed in (42, 43, 44)]
    _launch_shard("amf", "bdc-kroll-amf-20260927", run_amf_sweep, tasks, shard)


@APP_FGW.local_entrypoint()
def launch_fgw_shard(shard: int = 0) -> None:
    tasks = [{"arm": arm, "k": k, "seed": 42} for k in K0 for arm in ("mb4", "mb8")]
    _launch_shard("fgw", "bdc-kroll-fgw-20260927", run_fgw_sweep, tasks, shard)


@APP_TEMI.local_entrypoint()
def launch_temi_shard(shard: int = 0) -> None:
    tasks = [{"representation": name, "k": k, "seed": 42}
             for k in K0 for name in ("dinov2", "radio", "siglip2", "msn", "franca", "fashionsiglip")]
    _launch_shard("temi", "bdc-kroll-temi-20260927", run_temi_sweep, tasks, shard)


@APP_TAC.local_entrypoint()
def launch_tac_shard(shard: int = 0) -> None:
    tasks = [{"representation": name, "k": k}
             for k in K0 for name in ("siglip2", "radio_siglip2g")]
    _launch_shard("tac", "bdc-kroll-tac-20260927", run_tac_sweep, tasks, shard)


@APP_TURTLE.local_entrypoint()
def launch_turtle_shard(shard: int = 0) -> None:
    tasks = [{"arm": arm, "k": k, "seed": 42} for k in K0 for arm in ("mb4", "mb8")]
    tasks += [{"arm": arm, "k": 12, "seed": seed} for arm in ("mb4", "mb8") for seed in (43, 44)]
    _launch_shard("turtle", "bdc-kroll-turtle-20260927", run_turtle_gpu, tasks, shard)


def _launch_refinement(family: str, app_name: str, runner) -> None:
    path = LOCAL_RUN / "refinement_requests.json"
    if not path.exists(): raise FileNotFoundError(f"stage-1 requests missing: {path}")
    tasks = json.loads(path.read_text(encoding="utf-8")).get(family, [])
    if not tasks:
        print(f"REFINEMENT family={family} status=NO_TASKS", flush=True)
        return
    _record_call(f"{family}_refinement", app_name, runner.spawn(tasks), "refinement")


@APP_CLASSIC.local_entrypoint()
def launch_classic_refinement() -> None:
    _launch_refinement("classic", "bdc-kroll-classic-20260927", run_classic)


@APP_MGE.local_entrypoint()
def launch_mge_refinement() -> None:
    _launch_refinement("mge", "bdc-kroll-mge-20260927", run_mge_sweep)


@APP_MCSF.local_entrypoint()
def launch_mcsf_refinement() -> None:
    _launch_refinement("mcsf", "bdc-kroll-mcsf-20260927", run_mcsf_sweep)


@APP_AMF.local_entrypoint()
def launch_amf_refinement() -> None:
    _launch_refinement("amf", "bdc-kroll-amf-20260927", run_amf_sweep)


@APP_FGW.local_entrypoint()
def launch_fgw_refinement() -> None:
    _launch_refinement("fgw", "bdc-kroll-fgw-20260927", run_fgw_sweep)


@APP_TURTLE.local_entrypoint()
def launch_turtle_refinement() -> None:
    _launch_refinement("turtle", "bdc-kroll-turtle-20260927", run_turtle_gpu)


@APP_TEMI.local_entrypoint()
def launch_temi_refinement() -> None:
    _launch_refinement("temi", "bdc-kroll-temi-20260927", run_temi_sweep)


@APP_TAC.local_entrypoint()
def launch_tac_refinement() -> None:
    _launch_refinement("tac", "bdc-kroll-tac-20260927", run_tac_sweep)
