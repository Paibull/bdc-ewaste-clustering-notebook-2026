from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[2]
SEMI = ROOT / "semifinal"
OUT = SEMI / "results/final_claim_checks"
SEED = 20260927
BACKBONES = ("dinov3", "radio", "aimv2", "siglip2", "dinov2", "siglip", "convnext", "eva02")
FAMILIES = ("battery", "mobile", "pcb", "input_peripheral")
CLASS_TO_FAMILY = {
    "Battery Waste": "battery", "Mobile": "mobile", "PCB": "pcb",
    "Keyboard": "input_peripheral", "Mouse": "input_peripheral",
}
MATERIALS = ("plastic", "metal", "glass", "rubber", "paper_cardboard", "other_unknown")
KNOWN_MATERIALS = MATERIALS[:-1]
STATS = ("W", "H", "border_white", "white_frac", "sat_mean", "sat_std",
         "edge_mean", "colorfulness", "val_mean", "bg_uniform")
MAPS = {
    "fused_k12": {1: "battery", 5: "mobile", 8: "pcb", 0: "input_peripheral", 7: "input_peripheral"},
    "radio_k12": {4: "battery", 3: "mobile", 10: "pcb", 2: "input_peripheral", 11: "input_peripheral"},
    "fused_k14": {0: "battery", 5: "mobile", 12: "pcb", 6: "input_peripheral", 9: "input_peripheral"},
}


def ci(values):
    return [float(x) for x in np.quantile(np.asarray(values, dtype=float), [0.025, 0.975])]


def align_centers(saved, fitted, centers):
    k = len(np.unique(saved))
    overlap = np.zeros((k, k), dtype=np.int64)
    np.add.at(overlap, (saved.astype(int), fitted.astype(int)), 1)
    ref, fit = linear_sum_assignment(-overlap)
    reordered = np.empty_like(centers)
    for r, f in zip(ref, fit):
        reordered[r] = centers[f]
    return reordered


def family_index(names):
    return np.asarray([FAMILIES.index(CLASS_TO_FAMILY[x]) for x in names], dtype=np.int8)


def strict_metrics(y, pred):
    recall = {}
    for idx, family in enumerate(FAMILIES):
        recall[family] = float(np.mean(pred[y == idx] == idx))
    mapped = pred >= 0
    correct = pred == y
    return {
        "strict_micro_top1": float(correct.mean()),
        "strict_macro_top1": float(np.mean(list(recall.values()))),
        "mapped_coverage": float(mapped.mean()),
        "unmapped_rate": float((~mapped).mean()),
        "conditional_accuracy_mapped": float(correct[mapped].mean()) if mapped.any() else float("nan"),
        "recall_per_family": recall,
    }


def strict_eval(y, candidate_data):
    rng = np.random.default_rng(SEED)
    groups = [np.flatnonzero(y == i) for i in range(len(FAMILIES))]
    point = {name: strict_metrics(y, v["strict"]) for name, v in candidate_data.items()}
    for name, values in candidate_data.items():
        values["closed_point"] = float(np.mean([
            np.mean(values["closed"][y == i] == i) for i in range(len(FAMILIES))
        ]))
        point[name]["closed_set_macro_top1"] = values["closed_point"]
    draws = {name: {metric: np.empty(1000, dtype=float) for metric in
                    ("strict_micro_top1", "strict_macro_top1", "closed_set_macro_top1")}
             for name in candidate_data}
    deltas = {"fused_k12_minus_radio_k12": np.empty(1000),
              "fused_k14_minus_fused_k12": np.empty(1000)}
    for b in range(1000):
        ix = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        vals = {}
        for name, v in candidate_data.items():
            m = strict_metrics(y[ix], v["strict"][ix])
            cm = float(np.mean([np.mean(v["closed"][ix][y[ix] == i] == i) for i in range(4)]))
            vals[name] = m["strict_macro_top1"]
            for metric in ("strict_micro_top1", "strict_macro_top1"):
                draws[name][metric][b] = m[metric]
            draws[name]["closed_set_macro_top1"][b] = cm
        deltas["fused_k12_minus_radio_k12"][b] = vals["fused_k12"] - vals["radio_k12"]
        deltas["fused_k14_minus_fused_k12"][b] = vals["fused_k14"] - vals["fused_k12"]
    cis = {name: {metric: ci(draw) for metric, draw in metrics.items()} for name, metrics in draws.items()}
    delta_ci = {name: ci(values) for name, values in deltas.items()}
    null_rng = np.random.default_rng(SEED + 1)
    null = np.empty(999, dtype=float)
    observed = point["fused_k12"]["strict_macro_top1"]
    for i in range(len(null)):
        null[i] = strict_metrics(y, candidate_data["fused_k12"]["strict"][null_rng.permutation(len(y))])["strict_macro_top1"]
    return point, cis, delta_ci, null, float((1 + np.count_nonzero(null >= observed)) / 1000)


def normalize_sqrt(values):
    out = np.sqrt(np.maximum(values, 0.0))
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    zero = norms[:, 0] <= 1e-12
    out /= np.maximum(norms, 1e-12)
    if zero.any():
        out[zero] = 1.0 / np.sqrt(out.shape[1])
    return out


def normalize_root(values):
    out = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    zero = norms[:, 0] <= 1e-12
    out /= np.maximum(norms, 1e-12)
    if zero.any():
        out[zero] = 1.0 / np.sqrt(out.shape[1])
    return out


def source_stats_design(source_all, stats_all, discovery, holdout):
    cats = sorted(np.unique(source_all).tolist())
    if set(cats) != {"field", "web"}:
        raise ValueError(f"expected source categories web/field, got {cats}")
    tr_stats = stats_all[discovery]
    mu, sd = tr_stats.mean(axis=0), tr_stats.std(axis=0)
    sd[sd < 1e-12] = 1.0
    std_stats = (stats_all - mu) / sd
    source_dummy = (source_all == cats[1]).astype(np.float64)[:, None]
    baseline = np.column_stack((np.ones(len(source_all)), source_dummy, std_stats))
    return baseline[discovery], baseline[holdout], mu, sd


def cluster_design(labels, k):
    # K-1 indicators plus intercept give the same model space without dummy collinearity.
    return (np.asarray(labels, dtype=int)[:, None] == np.arange(1, k)).astype(np.float64)


def fit_predict_with_cluster(xtr, xte, ytr, train_labels, test_labels, k):
    ctrain, ctest = cluster_design(train_labels, k), cluster_design(test_labels, k)
    base_beta = np.linalg.lstsq(xtr, ytr, rcond=None)[0]
    base_pred = xte @ base_beta
    full_beta = np.linalg.lstsq(np.column_stack((xtr, ctrain)), ytr, rcond=None)[0]
    full_pred = np.column_stack((xte, ctest)) @ full_beta
    return base_pred, full_pred


def material_scores(ytrans, base_pred, full_pred):
    base_root, full_root = normalize_root(base_pred), normalize_root(full_pred)
    true_root = normalize_root(ytrans)
    base_fraction, full_fraction = base_root * base_root, full_root * full_root
    true_fraction = ytrans * ytrans
    # Hellinger distance is Euclidean distance between normalized square roots / sqrt(2).
    h_base = np.linalg.norm(true_root - base_root, axis=1) / np.sqrt(2.0)
    h_full = np.linalg.norm(true_root - full_root, axis=1) / np.sqrt(2.0)
    c_base = np.sum(true_root * base_root, axis=1)
    c_full = np.sum(true_root * full_root, axis=1)
    return {
        "true_fraction": true_fraction, "base_fraction": base_fraction, "full_fraction": full_fraction,
        "hellinger_base": h_base, "hellinger_full": h_full,
        "cosine_base": c_base, "cosine_full": c_full,
        "sse_base": np.square(ytrans - base_pred).sum(axis=1),
        "sse_full": np.square(ytrans - full_pred).sum(axis=1),
    }


def permute_within_source(values, source, rng):
    out = np.asarray(values).copy()
    for group in np.unique(source):
        ids = np.flatnonzero(source == group)
        out[ids] = rng.permutation(out[ids])
    return out


def permutation_incremental_r2(xtr, xte, ytr, yte, labels_tr, labels_te,
                               source_tr, source_te, k, observed_r2, rng, n_perm=999):
    # The base design is fixed; each null draw permutes cluster assignments independently within source.
    beta_base = np.linalg.lstsq(xtr, ytr, rcond=None)[0]
    pred_base = xte @ beta_base
    sse_base = np.square(yte - pred_base).sum()
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        lt = permute_within_source(labels_tr, source_tr, rng)
        le = permute_within_source(labels_te, source_te, rng)
        ctrain, ctest = cluster_design(lt, k), cluster_design(le, k)
        full_beta = np.linalg.lstsq(np.column_stack((xtr, ctrain)), ytr, rcond=None)[0]
        pred = np.column_stack((xte, ctest)) @ full_beta
        null[i] = 1.0 - np.square(yte - pred).sum() / max(sse_base, 1e-12)
    p = float((1 + np.count_nonzero(null >= observed_r2)) / (n_perm + 1))
    return p, float(np.quantile(null, 0.95))


def material_eval_case(name, k, labels_all, fractions, source, stats, discovery, holdout,
                       use_known_only=False, n_boot=1000):
    if use_known_only:
        mass = fractions[:, :-1].sum(axis=1)
        valid = mass > 1e-12
        yfrac = np.zeros((len(fractions), len(KNOWN_MATERIALS)), dtype=float)
        yfrac[valid] = fractions[valid, :-1] / mass[valid, None]
        train_mask = discovery & valid
        test_mask = holdout & valid
        material_names = KNOWN_MATERIALS
    else:
        valid = np.ones(len(fractions), dtype=bool)
        yfrac = fractions.copy()
        train_mask, test_mask = discovery, holdout
        material_names = MATERIALS
    ytrans = normalize_sqrt(yfrac[train_mask])
    ytest = normalize_sqrt(yfrac[test_mask])
    x_all_tr, x_all_te, _, _ = source_stats_design(source, stats, discovery, holdout)
    # Sensitivity excludes rows with no known-material mass from model fit/evaluation.
    xtr = x_all_tr[valid[discovery]]
    xte = x_all_te[valid[holdout]]
    src_tr, src_te = source[train_mask], source[test_mask]
    labels_tr, labels_te = labels_all[train_mask], labels_all[test_mask]
    base_pred, full_pred = fit_predict_with_cluster(xtr, xte, ytrans, labels_tr, labels_te, k)
    scored = material_scores(ytest, base_pred, full_pred)
    sse_base = float(scored["sse_base"].sum())
    sse_full = float(scored["sse_full"].sum())
    incremental = float(1.0 - sse_full / max(sse_base, 1e-12))
    h_reduction = float(np.mean(scored["hellinger_base"] - scored["hellinger_full"]))
    cosine_gain = float(np.mean(scored["cosine_full"] - scored["cosine_base"]))

    rng = np.random.default_rng(SEED + k + (100 if use_known_only else 0))
    source_groups = [np.flatnonzero(src_te == group) for group in sorted(np.unique(src_te))]
    boot = {key: np.empty(n_boot, dtype=float) for key in ("incremental_r2", "hellinger_reduction", "cosine_gain")}
    mat_boot = {mat: {"mae": np.empty(n_boot), "spearman": np.empty(n_boot)} for mat in material_names}
    for b in range(n_boot):
        ix = np.concatenate([rng.choice(g, len(g), replace=True) for g in source_groups])
        sb = float(1.0 - scored["sse_full"][ix].sum() / max(scored["sse_base"][ix].sum(), 1e-12))
        boot["incremental_r2"][b] = sb
        boot["hellinger_reduction"][b] = float(np.mean(scored["hellinger_base"][ix] - scored["hellinger_full"][ix]))
        boot["cosine_gain"][b] = float(np.mean(scored["cosine_full"][ix] - scored["cosine_base"][ix]))
        for j, mat in enumerate(material_names):
            actual, predicted = scored["true_fraction"][ix, j], scored["full_fraction"][ix, j]
            mat_boot[mat]["mae"][b] = float(np.mean(np.abs(actual - predicted)))
            mat_boot[mat]["spearman"][b] = float(spearmanr(actual, predicted).statistic) if np.ptp(actual) > 0 and np.ptp(predicted) > 0 else float("nan")

    p, null_q95 = permutation_incremental_r2(
        xtr, xte, ytrans, ytest, labels_tr, labels_te, src_tr, src_te,
        k, incremental, np.random.default_rng(SEED + 2000 + k + (100 if use_known_only else 0)),
    )
    metric = {
        "estimator": name, "target": "five_materials" if use_known_only else "six_materials",
        "n_discovery_fit": int(train_mask.sum()), "n_holdout": int(test_mask.sum()),
        "n_holdout_without_known_material_mass": int((holdout & ~valid).sum()) if use_known_only else 0,
        "incremental_r2": incremental, "incremental_r2_ci_low": ci(boot["incremental_r2"])[0],
        "incremental_r2_ci_high": ci(boot["incremental_r2"])[1], "within_source_permutation_p": p,
        "null_incremental_r2_q95": null_q95,
        "hellinger_baseline": float(np.mean(scored["hellinger_base"])),
        "hellinger_full": float(np.mean(scored["hellinger_full"])),
        "hellinger_reduction": h_reduction,
        "hellinger_reduction_ci_low": ci(boot["hellinger_reduction"])[0],
        "hellinger_reduction_ci_high": ci(boot["hellinger_reduction"])[1],
        "cosine_baseline": float(np.mean(scored["cosine_base"])),
        "cosine_full": float(np.mean(scored["cosine_full"])), "cosine_gain": cosine_gain,
        "cosine_gain_ci_low": ci(boot["cosine_gain"])[0], "cosine_gain_ci_high": ci(boot["cosine_gain"])[1],
        "bootstrap_draws": n_boot, "bootstrap_strata": "web/field",
    }
    per_material = []
    for j, mat in enumerate(material_names):
        actual, predicted = scored["true_fraction"][:, j], scored["full_fraction"][:, j]
        corr = float(spearmanr(actual, predicted).statistic) if np.ptp(actual) > 0 and np.ptp(predicted) > 0 else float("nan")
        per_material.append({
            "estimator": name, "target": metric["target"], "material": mat,
            "mae": float(np.mean(np.abs(actual - predicted))),
            "mae_ci_low": ci(mat_boot[mat]["mae"])[0], "mae_ci_high": ci(mat_boot[mat]["mae"])[1],
            "spearman": corr, "spearman_ci_low": ci(mat_boot[mat]["spearman"])[0],
            "spearman_ci_high": ci(mat_boot[mat]["spearman"])[1],
        })
    return metric, per_material


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(SEMI / "scripts"))
    from run_semantic_g0 import BACKBONES as G0_BACKBONES, assign_holdout, fit_pca_space, l2
    from run_bangladesh_frozen_transfer import align_reconstructed_centroids
    if tuple(G0_BACKBONES) != BACKBONES:
        raise ValueError("frozen backbone order mismatch")

    parts = np.load(SEMI / "artifacts/semantic_g0_partitions.npz", allow_pickle=False)
    keys = parts["keys"].astype(str)
    discovery, holdout = parts["discovery_mask"].astype(bool), parts["holdout_mask"].astype(bool)
    if len(keys) != 4179 or len(np.unique(keys)) != 4179 or int(discovery.sum()) != 3579 or int(holdout.sum()) != 600:
        raise ValueError("expected 4,179 unique keys split 3,579/600")
    if not np.array_equal(np.flatnonzero(discovery), np.flatnonzero(~holdout)):
        raise ValueError("discovery and holdout masks do not complement")

    subset = pd.read_csv(SEMI / "results/subset_validasi.csv", dtype={"key": str})
    if subset["key"].nunique() != 600 or set(subset["key"]) != set(keys[holdout]):
        raise ValueError("locked holdout keys do not match partition")

    ext = np.load(SEMI / "results/bangladesh_frozen_transfer/external_embeddings.npz", allow_pickle=False)
    ext_keys, classes = ext["key"].astype(str), ext["source_class"].astype(str)
    if len(ext_keys) != 2153 or len(np.unique(ext_keys)) != 2153:
        raise ValueError("expected 2,153 unique external keys")
    known_mask = np.asarray([name in CLASS_TO_FAMILY for name in classes])
    if int(known_mask.sum()) != 710:
        raise ValueError("expected 710 known external images")
    previous = pd.read_csv(SEMI / "results/bangladesh_frozen_transfer/predictions.csv", dtype={"key": str})
    if not np.array_equal(previous["key"].to_numpy(), ext_keys) or not np.array_equal(previous["source_class"].to_numpy(), classes):
        raise ValueError("external prediction key order differs from cached embeddings")
    old_metrics = json.loads((SEMI / "results/bangladesh_frozen_transfer/metrics.json").read_text(encoding="utf-8"))

    embeddings = np.load(SEMI / "artifacts/ewaste_emb_harm_resjpeg.npz", allow_pickle=False)
    if not np.array_equal(embeddings["key"].astype(str), keys):
        raise ValueError("embedding and partition key order mismatch")
    # K12's frozen partition uses semantic_g0.fit_pca_space (L2 after each PCA).
    # Cached K14 sweep labels were produced by evaluate_bangladesh_k14_sensitivity's
    # legacy fused_spaces constructor (L2 only after concatenating the PCA blocks).
    # Keep both existing constructors exact so each saved partition reconstructs.
    fused_train_blocks, fused_hold_blocks, fused_ext_blocks = [], [], []
    sweep_train_blocks, sweep_hold_blocks, sweep_ext_blocks = [], [], []
    radio_raw = None
    for backbone in BACKBONES:
        raw = np.asarray(embeddings[backbone], dtype=np.float32)
        e = np.asarray(ext[backbone], dtype=np.float32)
        if not np.isfinite(raw).all() or not np.isfinite(e).all():
            raise ValueError(f"non-finite values in {backbone}")
        train, test, pca = fit_pca_space(raw[discovery], raw[holdout], 32)
        fused_train_blocks.append(train)
        fused_hold_blocks.append(test)
        fused_ext_blocks.append(l2(pca.transform(l2(e))))
        sweep_pca = PCA(n_components=32, svd_solver="randomized", whiten=False, random_state=0)
        sweep_input = l2(raw[discovery])
        sweep_train_blocks.append(sweep_pca.fit_transform(sweep_input))
        sweep_hold_blocks.append(sweep_pca.transform(l2(raw[holdout])))
        sweep_ext_blocks.append(sweep_pca.transform(l2(e)))
        if backbone == "radio":
            radio_raw = raw
        del raw, e
    fused_train_join = np.concatenate(fused_train_blocks, axis=1)
    fused_hold_join = np.concatenate(fused_hold_blocks, axis=1)
    fused_ext_join = np.concatenate(fused_ext_blocks, axis=1)
    fused_train, fused_hold, fused_pca = fit_pca_space(fused_train_join, fused_hold_join, 100)
    fused_external = l2(fused_pca.transform(l2(fused_ext_join)))
    sweep_train_join = l2(np.concatenate(sweep_train_blocks, axis=1))
    sweep_hold_join = l2(np.concatenate(sweep_hold_blocks, axis=1))
    sweep_ext_join = l2(np.concatenate(sweep_ext_blocks, axis=1))
    sweep_pca = PCA(n_components=100, svd_solver="randomized", whiten=False, random_state=0)
    fused14_train = l2(sweep_pca.fit_transform(sweep_train_join))
    fused14_hold = l2(sweep_pca.transform(sweep_hold_join))
    fused14_external = l2(sweep_pca.transform(sweep_ext_join))
    radio_train, radio_hold, radio_pca = fit_pca_space(radio_raw[discovery], radio_raw[holdout], 100)
    radio_external = l2(radio_pca.transform(l2(np.asarray(ext["radio"], dtype=np.float32))))
    if not all(np.isfinite(v).all() for v in (fused_train, fused_hold, fused_external, fused14_train, fused14_hold,
                                               fused14_external, radio_train, radio_hold, radio_external)):
        raise ValueError("non-finite reconstructed spaces")

    saved14 = np.load(SEMI / "results/k_resolution_sweep/20260927/remote/classic/kmeans_fused/k_14/seed_42/labels.npz", allow_pickle=False)
    if not np.array_equal(saved14["key"].astype(str), keys[discovery]):
        raise ValueError("saved K14 discovery key order mismatch")
    saved_k14 = saved14["labels"].astype(np.int32)
    recon, centers, checks = {}, {}, {}
    for name, space, k, seed, frozen in (
        ("fused_k12", fused_train, 12, 0, parts["labels__kmeans_fused"][discovery].astype(np.int32)),
        ("radio_k12", radio_train, 12, 0, parts["labels__kmeans_radio"][discovery].astype(np.int32)),
        ("fused_k14", fused14_train, 14, 42, saved_k14),
    ):
        model = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(space)
        ari = float(adjusted_rand_score(frozen, model.labels_))
        checks[name] = ari
        if ari != 1.0:
            raise RuntimeError(f"reconstruction ARI failed for {name}: {ari:.10f}")
        if name == "fused_k12":
            _, mapping, aligned = align_reconstructed_centroids(frozen, model.labels_, model.cluster_centers_)
            centers[name] = aligned
        else:
            centers[name] = align_centers(frozen, model.labels_, model.cluster_centers_)
        recon[name] = frozen
        print(f"Reconstruction {name}: ARI={ari:.4f}", flush=True)

    labels12 = parts["labels__kmeans_fused"].astype(np.int32)
    labels14_all = np.empty(len(keys), dtype=np.int32)
    labels14_all[discovery] = saved_k14
    labels14_all[holdout] = assign_holdout(fused14_train, fused14_hold, saved_k14)[0]

    y = family_index(classes[known_mask])
    strict_rows, pred_arrays = [], {}
    model_specs = (("fused_k12", fused_external, MAPS["fused_k12"], 12),
                   ("radio_k12", radio_external, MAPS["radio_k12"], 12),
                   ("fused_k14", fused14_external, MAPS["fused_k14"], 14))
    for name, outside, mapping, k in model_specs:
        dist = 1.0 - l2(outside[known_mask]) @ l2(centers[name]).T
        nearest = np.argmin(dist, axis=1)
        strict_pred = np.asarray([FAMILIES.index(mapping[int(c)]) if int(c) in mapping else -1 for c in nearest], dtype=np.int8)
        family_dist = np.column_stack([
            np.min(dist[:, [cluster for cluster, family in mapping.items() if family == fam]], axis=1)
            for fam in FAMILIES
        ])
        closed_pred = np.argmin(family_dist, axis=1).astype(np.int8)
        pred_arrays[name] = {"strict": strict_pred, "closed": closed_pred, "nearest_cluster": nearest,
                             "distance": dist.min(axis=1)}
    print("Scoring strict transfer: bootstrap 1,000 + null permutation 999", flush=True)
    points, cis, delta_cis, null, null_p = strict_eval(y, pred_arrays)
    null_q95 = float(np.quantile(null, 0.95))
    strict_verdict = "STRICT_TRANSFER_SUPPORTED" if cis["fused_k12"]["strict_macro_top1"][0] > null_q95 else (
        "CLOSED_SET_ONLY" if cis["fused_k12"]["closed_set_macro_top1"][0] > 0.25 else "STRICT_TRANSFER_NOT_SUPPORTED")
    strict_rows = []
    for name in pred_arrays:
        row = {"estimator": name, **points[name]}
        for metric, limits in cis[name].items():
            row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = limits
        if name == "fused_k12":
            row["strict_macro_null_q95"] = null_q95
            row["strict_macro_permutation_p"] = null_p
        strict_rows.append(row)
    for name, vals in delta_cis.items():
        strict_rows.append({"estimator": name, "metric": "paired_delta_strict_macro_top1",
                            "ci_low": vals[0], "ci_high": vals[1]})

    # Material descriptor regression on locked holdout, reusing the same reconstructed fused spaces.
    frac = pd.read_csv(SEMI / "results/dms46_full/fraction_table.csv", dtype={"key": str})
    if frac["key"].nunique() != 4179 or set(frac["key"]) != set(keys):
        raise ValueError("DMS fraction keys do not match the 4,179 locked examples")
    frac = frac.set_index("key").loc[keys]
    fractions = frac.loc[:, MATERIALS].to_numpy(dtype=float)
    if not np.isfinite(fractions).all() or not np.allclose(fractions.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("six DMS fractions must be finite and sum to one")
    source = frac["source"].astype(str).to_numpy()
    if set(np.unique(source)) != {"web", "field"}:
        raise ValueError("material source must identify web/field")
    imgstats = pd.read_csv(SEMI / "results/imgstats.csv", dtype={"key": str})
    if imgstats["key"].nunique() != 4179 or set(imgstats["key"]) != set(keys):
        raise ValueError("imgstats keys do not match the 4,179 locked examples")
    imgstats = imgstats.set_index("key").loc[keys]
    stats = imgstats.loc[:, STATS].to_numpy(dtype=float)
    stats[:, 0:2] = np.log1p(stats[:, 0:2])
    if not np.isfinite(stats).all():
        raise ValueError("non-finite acquisition/background statistics")

    material_rows, per_material_rows = [], []
    for name, k, label_vector in (("fused_k12", 12, labels12), ("fused_k14", 14, labels14_all)):
        for known_only in (False, True):
            target_name = "five-material sensitivity" if known_only else "six-material primary"
            print(f"Material holdout {name}: {target_name}, bootstrap 1,000 + permutation 999", flush=True)
            metric, by_material = material_eval_case(
                name, k, label_vector, fractions, source, stats, discovery, holdout, known_only,
            )
            material_rows.append(metric)
            per_material_rows.extend(by_material)
    primary = {row["estimator"]: row for row in material_rows if row["target"] == "six_materials"}
    sensitivity = {row["estimator"]: row for row in material_rows if row["target"] == "five_materials"}
    primary_pass = {name: primary[name]["incremental_r2_ci_low"] > 0 and
                    primary[name]["within_source_permutation_p"] <= 0.05 for name in primary}
    sensitivity_pass = {name: sensitivity[name]["incremental_r2_ci_low"] > 0 and
                        sensitivity[name]["within_source_permutation_p"] <= 0.05 for name in sensitivity}
    if any(row["incremental_r2"] > 0 for row in primary.values()) and not all(sensitivity_pass.values()):
        material_verdict = "DRIVEN_BY_OTHER_UNKNOWN"
    elif all(primary_pass.values()):
        material_verdict = "ROBUST_MATERIAL_SUPPORT"
    elif sum(primary_pass.values()) == 1:
        material_verdict = "RESOLUTION_SENSITIVE"
    else:
        material_verdict = "NO_HOLDOUT_SUPPORT"

    pd.DataFrame(strict_rows).to_csv(OUT / "strict_transfer_metrics.csv", index=False, encoding="utf-8")
    prediction_rows = []
    keys_ext = ext_keys[known_mask]
    source_ext = classes[known_mask]
    for i, key in enumerate(keys_ext):
        row = {"key": key, "source_class": source_ext[i], "true_family": FAMILIES[int(y[i])],
               "nearest_cluster_fused_k12": int(pred_arrays["fused_k12"]["nearest_cluster"][i]),
               "nearest_cluster_radio_k12": int(pred_arrays["radio_k12"]["nearest_cluster"][i]),
               "nearest_cluster_fused_k14": int(pred_arrays["fused_k14"]["nearest_cluster"][i])}
        for name in pred_arrays:
            p = int(pred_arrays[name]["strict"][i])
            row[f"strict_prediction_{name}"] = FAMILIES[p] if p >= 0 else "unmapped"
            row[f"nearest_distance_{name}"] = float(pred_arrays[name]["distance"][i])
            row[f"closed_prediction_{name}"] = FAMILIES[int(pred_arrays[name]["closed"][i])]
        prediction_rows.append(row)
    pd.DataFrame(prediction_rows).to_csv(OUT / "strict_transfer_predictions.csv", index=False, encoding="utf-8")
    pd.DataFrame(material_rows).to_csv(OUT / "material_holdout_metrics.csv", index=False, encoding="utf-8")
    pd.DataFrame(per_material_rows).to_csv(OUT / "material_per_material.csv", index=False, encoding="utf-8")

    old_closed = old_metrics.get("primary_known_set", {}).get("estimators", {})
    summary = {
        "validation": {
            "bdc_keys": int(len(keys)), "discovery": int(discovery.sum()), "holdout": int(holdout.sum()),
            "external_keys": int(len(ext_keys)), "known_external": int(known_mask.sum()),
            "holdout_subset_key_match": True, "finite_reconstruction": True,
            "dms_six_fraction_sum_tolerance": 1e-5,
        },
        "reconstruction_ari": checks,
        "strict_transfer": {
            "frozen_mappings": {name: {str(k): v for k, v in mapping.items()} for name, mapping in MAPS.items()},
            "point_metrics": points, "bootstrap_ci_95": cis,
            "paired_delta_ci_95": delta_cis, "bootstrap_draws": 1000,
            "bootstrap_strata": "true family", "permutation_draws": 999,
            "permutation_null": "shuffle strict predictions across rows, preserving prediction/unmapped counts",
            "null_macro_q95": null_q95, "null_macro_p_for_fused_k12": null_p,
            "verdict": strict_verdict,
            "fusion_strict_gain": bool(delta_cis["fused_k12_minus_radio_k12"][0] > 0),
            "k12_k14_transfer_not_sensitive": bool(delta_cis["fused_k14_minus_fused_k12"][0] <= 0 <= delta_cis["fused_k14_minus_fused_k12"][1]),
            "previous_closed_set_metrics": {k: old_closed.get(k) for k in ("fused", "radio")},
        },
        "material_holdout": {
            "fit_rows": int(discovery.sum()), "holdout_rows": int(holdout.sum()),
            "target_transform": "sqrt(fraction), row L2 normalization",
            "baseline": "source + discovery-standardized acquisition/background statistics",
            "full": "baseline + one-hot cluster (reference-coded)",
            "bootstrap_draws": 1000, "bootstrap_strata": "web/field",
            "permutation_draws": 999, "permutation": "cluster labels independently shuffled within web/field in fit and holdout",
            "n_holdout_without_known_material_mass": int((holdout & (fractions[:, :-1].sum(axis=1) <= 1e-12)).sum()),
            "metrics": material_rows, "primary_support_by_k": primary_pass,
            "five_material_sensitivity_support_by_k": sensitivity_pass, "verdict": material_verdict,
            "ami_hard_k6_diagnostic_only": "not recomputed or used for verdict",
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    write_report(summary)
    print(json.dumps({"reconstruction_ari": checks, "strict_verdict": strict_verdict,
                      "material_verdict": material_verdict, "runtime_seconds": summary["runtime_seconds"]}, ensure_ascii=False))


def write_report(s):
    strict = s["strict_transfer"]
    mat = s["material_holdout"]
    rows = strict["point_metrics"]
    t = ["# Strict transfer dan material holdout", "",
         "## A. Strict frozen external transfer", "",
         "Nearest cluster dicari di seluruh centroid; cluster tanpa mapping dihitung sebagai salah. Mapping BDC tetap dibekukan.",
         "K12 menggunakan transform beku `run_semantic_g0`; label K14 berasal dari sweep lama dengan constructor PCA/fusion berbeda dan direkonstruksi melalui transform sensitivitas K14 yang tersimpan. Keduanya lolos ARI 1.0000, tetapi selisih K14−K12 tidak dapat diatribusikan ke jumlah cluster saja.", "",
         "| Estimator | Strict micro | Strict macro | 95% CI macro | Coverage mapped | Unmapped | Conditional accuracy | Closed-set macro |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("fused_k12", "radio_k12", "fused_k14"):
        r = rows[name]; lo, hi = strict["bootstrap_ci_95"][name]["strict_macro_top1"]
        t.append(f"| {name} | {r['strict_micro_top1']:.3f} | {r['strict_macro_top1']:.3f} | [{lo:.3f}, {hi:.3f}] | {r['mapped_coverage']:.3f} | {r['unmapped_rate']:.3f} | {r['conditional_accuracy_mapped']:.3f} | {r['closed_set_macro_top1']:.3f} |")
    for name in ("fused_k12", "radio_k12", "fused_k14"):
        r = rows[name]["recall_per_family"]
        t.append(f"Recall {name} (battery / mobile / PCB / input peripheral): " +
                 " / ".join(f"{r[x]:.3f}" for x in FAMILIES) + ".")
    d12, d14 = strict["paired_delta_ci_95"]["fused_k12_minus_radio_k12"], strict["paired_delta_ci_95"]["fused_k14_minus_fused_k12"]
    t += ["", f"Strict macro null q95: {strict['null_macro_q95']:.3f}; fused K12 permutation p={strict['null_macro_p_for_fused_k12']:.4f}.",
          f"Paired delta fused K12−RADIO K12: [{d12[0]:.3f}, {d12[1]:.3f}]; fused K14−K12: [{d14[0]:.3f}, {d14[1]:.3f}].",
          f"Verdict A: **{strict['verdict']}**. Fusion strict gain: **{strict['fusion_strict_gain']}**. " +
          ("CI delta K14−K12 memuat nol." if strict["k12_k14_transfer_not_sensitive"] else
           "CI delta K14−K12 tidak memuat nol, tetapi karena constructor space berbeda, ini bukan bukti bahwa perubahan K saja meningkatkan transfer."),
          "", "## B. Continuous material pada locked holdout", "",
          "Target ditransformasi sebagai sqrt(fraction) lalu dinormalisasi L2. Baseline memakai source dan statistik akuisisi/background yang distandardisasi dari discovery; full menambahkan indikator cluster. CI memakai bootstrap web/field dan p memakai permutasi cluster dalam source.", "",
          "| Partisi | Target | Incremental R² (95% CI) | Δ Hellinger (95% CI) | Δ cosine (95% CI) | Permutation p | N holdout |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for r in mat["metrics"]:
        t.append(f"| {r['estimator']} | {r['target']} | {r['incremental_r2']:.4f} [{r['incremental_r2_ci_low']:.4f}, {r['incremental_r2_ci_high']:.4f}] | {r['hellinger_reduction']:.4f} [{r['hellinger_reduction_ci_low']:.4f}, {r['hellinger_reduction_ci_high']:.4f}] | {r['cosine_gain']:.4f} [{r['cosine_gain_ci_low']:.4f}, {r['cosine_gain_ci_high']:.4f}] | {r['within_source_permutation_p']:.4f} | {r['n_holdout']} |")
    t += ["", f"Baris holdout tanpa mass pada lima material known: {mat['n_holdout_without_known_material_mass']}. Verdict B: **{mat['verdict']}**.",
          "", "## Kesimpulan untuk paper", "",
          f"Reconstruction ARI: {', '.join(f'{k}={v:.4f}' for k, v in s['reconstruction_ari'].items())}. Klaim transfer harus mengikuti hasil nearest centroid global dan tingkat cluster unmapped; skor closed-set saja tidak membuktikan setiap gambar dapat dipetakan oleh taxonomy beku.",
          "Klaim aman: cluster membawa informasi tambahan tentang komposisi DMS pada locked holdout setelah source dan covariates akuisisi/background dikontrol. Klaim harus dipersempit: ini bukan validasi pixel-level material ground truth atau bukti bahwa cluster adalah taxonomy material; efek K12/K14 tidak dapat dipisahkan dari perbedaan constructor fused yang diwarisi.",
          "", f"Runtime scoring: {s['runtime_seconds'] / 60:.1f} menit. " ]
    report = SEMI / "docs/114-HASIL-STRICT-TRANSFER-MATERIAL-HOLDOUT.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(t) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
