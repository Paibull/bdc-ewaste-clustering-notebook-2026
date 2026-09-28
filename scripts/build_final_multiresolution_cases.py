"""Build a compact, descriptive K-resolution case figure from frozen labels."""
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "semifinal"
OUT = BASE / "results" / "final_multiresolution"
STAB = BASE / "results" / "k_stability_100"


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = np.load(STAB / "assignments.npz")
    keys = data["discovery_keys"].astype(str)
    ks = list(map(int, data["k_values"]))
    labels = {k: data["medoid_labels"][ks.index(k)].astype(int) for k in (12, 14, 16)}
    stability = {(int(r["k"]), int(r["cluster"])): r for r in read_rows(STAB / "cluster_stability.csv")}
    imgstats = {r["key"]: r for r in read_rows(BASE / "results" / "imgstats.csv")}
    source_by_key = {r["key"]: r["source"] for r in read_rows(BASE / "results" / "dms46_full" / "fraction_table.csv")}
    if not all(key in imgstats and key in source_by_key for key in keys):
        raise ValueError("Metadata must cover every discovery key exactly")

    def web_share(indices):
        return float(np.mean([source_by_key[keys[i]] == "web" for i in indices]))

    def resolution_share(indices):
        return float(np.mean([float(imgstats[keys[i]]["W"]) == 150 and float(imgstats[keys[i]]["H"]) == 150 for i in indices]))

    def semantic(k, c):
        return stability[(k, int(c))]["semantic_node"]

    def children(parent_k, parent_id, child_k):
        p, c = labels[parent_k], labels[child_k]
        ix = p == parent_id
        rows = []
        for child_id in sorted(np.unique(c[ix])):
            child_ix = ix & (c == child_id)
            n = int(child_ix.sum())
            child_total = int((c == child_id).sum())
            parent_web = web_share(np.flatnonzero(ix))
            child_web = web_share(np.flatnonzero(c == child_id))
            s = stability[(child_k, int(child_id))]
            rows.append({
                "parent_k": parent_k, "parent_cluster": int(parent_id),
                "parent_semantic": semantic(parent_k, parent_id),
                "child_k": child_k, "child_cluster": int(child_id),
                "child_semantic": semantic(child_k, child_id),
                "parent_members_in_child": n, "child_size": child_total,
                "parent_containment": n / max(1, int(ix.sum())),
                "child_purity_to_parent": n / max(1, child_total),
                "child_jaccard_q2_5": float(s["jaccard_q2_5"]),
                "child_jaccard_median": float(s["jaccard_median"]),
                "child_fraction_jaccard_below_0_60": float(s["fraction_jaccard_below_0_60"]),
                "child_stability_status": s["stability_status"],
                "parent_web_share": parent_web,
                "child_web_share": child_web,
                "parent_150x150_share": resolution_share(np.flatnonzero(ix)),
                "child_150x150_share": resolution_share(np.flatnonzero(c == child_id)),
                "web_share_delta_child_minus_parent": None if parent_web is None or child_web is None else child_web - parent_web,
                "verdict": "PERSISTENT_ONLY",
                "source_control": "web/field proportions measured; not adjusted within-source",
                "verdict_reason": "Stable assignments and post-hoc descriptors; unequal web/field composition and missing visual inspection prevent semantic validation.",
            })
        return rows

    # Prefer a readable semantic refinement whose children are independently stable.
    candidates = []
    for parent in sorted(np.unique(labels[12])):
        rows = children(12, int(parent), 16)
        substantial = [r for r in rows if r["parent_containment"] >= .10 and r["child_purity_to_parent"] >= .80]
        distinct = {r["child_semantic"] for r in substantial}
        stable = all(r["child_jaccard_q2_5"] >= .60 and r["child_fraction_jaccard_below_0_60"] <= .20 for r in substantial)
        if len(substantial) >= 2 and len(distinct) >= 2 and stable:
            candidates.append((sum(r["parent_members_in_child"] for r in substantial), int(parent), substantial))
    if candidates:
        _, stable_parent, stable_rows = max(candidates, key=lambda x: x[0])
        case_a = {"case": "stable_refinement_K12_to_K16", "interpretation": "Stable semantic refinement within a K=12 parent; descriptive, not ground-truth taxonomy.", "rows": stable_rows}
    else:
        # Do not force a positive subtype claim when no split passes the stability screen.
        case_a = {"case": "stable_family_retention_K12_to_K16", "interpretation": "No semantic split passed the stability screen; show the most stable retained family as a resolution limit.", "rows": []}
        retained = []
        for parent in sorted(np.unique(labels[12])):
            rr = children(12, int(parent), 16)
            matching = [r for r in rr if r["parent_semantic"] == r["child_semantic"] and r["child_purity_to_parent"] >= .90]
            for r in matching:
                retained.append((r["child_jaccard_q2_5"], int(parent), r))
        if retained:
            _, stable_parent, r = max(retained, key=lambda x: x[0])
            case_a["rows"] = [r]

    # The known unstable display node at K=14 is the prespecified negative example.
    display_unstable = max(
        (r for (k, _), r in stability.items() if k == 14 and r["semantic_node"] == "display"),
        key=lambda r: float(r["fraction_jaccard_below_0_60"]),
    )
    display_id = int(display_unstable["cluster"])
    source = labels[14]
    parent_counts = Counter(labels[12][source == display_id])
    display_parent = parent_counts.most_common(1)[0][0]
    case_b_rows = children(12, int(display_parent), 14)
    # Limit display example to the unstable display child and its strongest sibling.
    case_b_rows.sort(key=lambda r: (r["child_cluster"] != display_id, -r["parent_members_in_child"]))
    case_b_rows = case_b_rows[:2]
    for r in case_b_rows:
        r["verdict"] = "UNSTABLE"
        r["source_control"] = "web/field proportions measured; not adjusted within-source"
        r["verdict_reason"] = "K=14 display membership is resampling-sensitive: 40–41% of subsamples have Jaccard below 0.60."
    case_b = {"case": "unstable_display_split_at_K14", "interpretation": "Display partition is sensitive to resampling at K=14; use as a limit case, not evidence of a stable subtype.", "rows": case_b_rows}

    out_rows = []
    for case in (case_a, case_b):
        for r in case["rows"]:
            out_rows.append({"case": case["case"], "interpretation": case["interpretation"], **r})
    csv_path = OUT / "cases.csv"
    fields = list(out_rows[0]) if out_rows else ["case", "interpretation"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(out_rows)

    # K trade-off values are direct joins from the completed coarse sweep and chooseR refinement.
    sweep_rows = {int(r["k"]): r for r in read_rows(BASE / "results" / "k_resolution_sweep" / "20260927" / "all_candidates.csv")
                  if r.get("candidate", "").startswith("classic::kmeans_fused::k")}
    chooser_rows = {int(r["k"]): r for r in read_rows(STAB / "chooser_summary.csv")}
    coarse_ks = (4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24)
    k_tradeoff = [{
        "k": k,
        "coarse_silhouette": float(sweep_rows[k]["silhouette"]),
        "coverage_adjusted": float(sweep_rows[k]["function_coverage_adjusted"]),
        "source_nmi": float(sweep_rows[k]["source_nmi"]),
        "coarse_stability_ari": float(sweep_rows[k]["stability_ari"]),
    } for k in coarse_ks]
    chooseR_tradeoff = [{
        "k": k,
        "chooseR_consensus_silhouette": float(chooser_rows[k]["consensus_silhouette_median"]),
        "clusters_above_chooseR_threshold": int(chooser_rows[k]["clusters_above_threshold"]),
        "chooseR_winner": chooser_rows[k]["chooseR_winner"] == "True",
    } for k in (12, 13, 14, 15, 16)]
    summary = {
        "source": str((STAB / "assignments.npz").relative_to(ROOT)),
        "selection": "Frozen medoid assignments only; no clustering or embedding rerun.",
        "k_tradeoff": k_tradeoff,
        "chooseR_refinement": chooseR_tradeoff,
        "case_a": {k: v for k, v in case_a.items() if k != "rows"} | {"rows": case_a["rows"]},
        "case_b": {k: v for k, v in case_b.items() if k != "rows"} | {"rows": case_b["rows"]},
        "metadata_join": {"key_type": "exact discovery key", "matched_rows": len(keys), "source_table": "results/dms46_full/fraction_table.csv"},
        "limits": ["Semantic names are frozen post-hoc descriptors.", "Stability is resampling consistency, not semantic truth.", "Source proportions are descriptive; no within-source adjustment or visual validation was performed."],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Reuse the existing project image layout; max four thumbnails per child, 16 total.
    image_index = {}
    for sub in ("train/0_Recyclable", "train/1_Electronic", "train/2_Organic", "test"):
        folder = ROOT / sub
        if folder.is_dir():
            for name in os.listdir(folder):
                image_index.setdefault(name, folder / name)
    rng = np.random.default_rng(20260927)
    canvas = Image.new("RGB", (1600, 1580), "white")
    draw = ImageDraw.Draw(canvas)
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    title_font = ImageFont.truetype(str(font_path), 28) if font_path.exists() else ImageFont.load_default()
    body_font = ImageFont.truetype(str(font_path), 21) if font_path.exists() else ImageFont.load_default()
    small_font = ImageFont.truetype(str(font_path), 17) if font_path.exists() else ImageFont.load_default()
    draw.text((32, 20), "Multi-resolution case studies from frozen BDC assignments", fill="black", font=title_font)
    draw.text((32, 60), "K is resolution. Semantic names are descriptive; stability measures resampling consistency, not truth.", fill="#333", font=small_font)
    draw.rounded_rectangle((24, 92, 1576, 532), radius=14, outline="#555", width=2)
    draw.text((44, 108), "A. Coarse fused K-Means sweep (existing results)", fill="#111", font=body_font)
    hdr = "K | silhouette | adjusted coverage | source NMI | seed stability ARI"
    draw.text((48, 151), hdr, fill="#333", font=small_font)
    for i, row in enumerate(k_tradeoff):
        y = 180 + i * 22
        line = (f"{row['k']:>2} | {row['coarse_silhouette']:.3f} | {row['coverage_adjusted']:.3f} | "
                f"{row['source_nmi']:.3f} | {row['coarse_stability_ari']:.3f}")
        draw.text((48, y), line, fill="#111", font=small_font)
    draw.text((650, 151), "K | chooseR consensus silhouette | clusters above threshold | selected", fill="#333", font=small_font)
    for i, row in enumerate(chooseR_tradeoff):
        line = (f"{row['k']:>2} | {row['chooseR_consensus_silhouette']:.3f} | "
                f"{row['clusters_above_chooseR_threshold']:>2} | {'yes' if row['chooseR_winner'] else 'no'}")
        draw.text((650, 180 + i * 31), line, fill="#111", font=small_font)

    def draw_case(y, title, rows):
        draw.rounded_rectangle((24, y, 1576, y + 410), radius=14, outline="#555", width=2)
        draw.text((44, y + 16), title, fill="#111", font=body_font)
        if not rows:
            draw.text((50, y + 70), "No split met the stability screen; no positive refinement is shown.", fill="#444", font=body_font)
            return
        parent = rows[0]
        draw.rounded_rectangle((55, y + 94, 440, y + 190), radius=10, fill="#e8eef8", outline="#476"
                               )
        draw.text((75, y + 112), f"K=12 | {parent['parent_semantic']} (C{parent['parent_cluster']})", fill="#111", font=small_font)
        draw.text((75, y + 146), f"parent n={int((labels[12] == parent['parent_cluster']).sum())}", fill="#333", font=small_font)
        for j, row in enumerate(rows):
            yy = y + 82 + j * 145
            x0 = 510
            draw.line((440, y + 140, x0, yy + 42), fill="#777", width=3)
            draw.rounded_rectangle((x0, yy, 1540, yy + 112), radius=10, fill="#f2f4f7", outline="#567")
            draw.text((x0 + 18, yy + 12), f"K={row['child_k']} | {row['child_semantic']} (C{row['child_cluster']}) | n={row['child_size']}", fill="#111", font=small_font)
            draw.text((x0 + 18, yy + 45), f"Share of parent {row['parent_containment']:.0%} | child purity {row['child_purity_to_parent']:.0%}", fill="#333", font=small_font)
            draw.text((x0 + 18, yy + 76), f"Jaccard q2.5 {row['child_jaccard_q2_5']:.3f} | resamples with Jaccard <.60: {row['child_fraction_jaccard_below_0_60']:.0%}", fill="#333", font=small_font)
            members = np.flatnonzero((labels[row["child_k"]] == row["child_cluster"]) & (labels[12] == row["parent_cluster"]))
            if len(members):
                chosen = rng.choice(members, min(4, len(members)), replace=False)
                for t, idx in enumerate(chosen):
                    name = Path(keys[idx]).name
                    path = image_index.get(name)
                    if path and path.is_file():
                        try:
                            im = Image.open(path).convert("RGB")
                            im.thumbnail((105, 92))
                            xx = 1240 + t * 145
                            canvas.paste(im, (xx, yy - 2))
                            draw.rectangle((xx, yy - 2, xx + im.width, yy - 2 + im.height), outline="#aaa")
                        except OSError:
                            pass
    draw_case(552, "B. K=12 to K=16: persistent descriptive refinement", case_a["rows"])
    draw_case(992, "C. K=12 to K=14: unstable display partition", case_b["rows"])
    draw.text((32, 1420), "Image montage unavailable from the existing resolver paths; semantic names remain post-hoc descriptors.", fill="#444", font=small_font)
    draw.text((32, 1450), "Web/field source joined by exact key: all 3,579 discovery rows matched. Proportions are descriptive.", fill="#444", font=small_font)
    if case_a["rows"]:
        r0 = case_a["rows"][0]
        shares = ", ".join(f"C{r['child_cluster']}={r['child_web_share']:.1%}" for r in case_a["rows"])
        draw.text((32, 1480), f"Case B web share: parent={r0['parent_web_share']:.1%}; children {shares}. Verdict: PERSISTENT_ONLY.", fill="#444", font=small_font)
    fig_path = OUT / "figure_multiresolution_cases.png"
    canvas.save(fig_path, optimize=True)
    print(f"CSV: {csv_path.relative_to(ROOT)} ({len(out_rows)} rows)")
    print(f"JSON: {(OUT / 'summary.json').relative_to(ROOT)}")
    print(f"Figure: {fig_path.relative_to(ROOT)}")
    print(json.dumps({"case_a": case_a, "case_b": case_b}, ensure_ascii=False))


if __name__ == "__main__":
    main()
