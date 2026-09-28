"""Small, runnable checks for the paper's main numerical results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean


class EvidenceRepository:
    def __init__(self, root: Path):
        self.tables = root / "tables"

    def csv(self, relative_path: str) -> list[dict[str, str]]:
        with (self.tables / relative_path).open(encoding="utf-8-sig", newline="") as source:
            return list(csv.DictReader(source))

    def json(self, relative_path: str) -> dict:
        with (self.tables / relative_path).open(encoding="utf-8") as source:
            return json.load(source)

    def selected_k(self) -> dict:
        rows = self.csv("k_stability_100/chooser_summary.csv")
        winners = [row for row in rows if row["chooseR_winner"] == "True"]
        if len(winners) != 1:
            raise ValueError("Expected one chooseR winner")
        k = int(winners[0]["k"])
        seeds = self.csv("k_stability_100/optimizer_stability.csv")
        ari = [float(row["ari_to_medoid"]) for row in seeds if int(row["k"]) == k]
        if len(ari) != 100:
            raise ValueError("Expected 100 seed results for the selected K")
        return {"k": k, "mean_ari_to_medoid": mean(ari), "seeds": len(ari)}

    def strict_transfer(self) -> dict:
        rows = self.csv("final_claim_checks/strict_transfer_predictions.csv")
        families = sorted({row["true_family"] for row in rows})
        if len(rows) != 710 or len(families) != 4:
            raise ValueError("Unexpected external evaluation population")
        result = {"n": len(rows), "classes": families}
        for model in ("fused_k12", "radio_k12"):
            column = f"strict_prediction_{model}"
            result[f"macro_top1_{model}"] = mean(
                sum(row[column] == family for row in rows if row["true_family"] == family)
                / sum(row["true_family"] == family for row in rows)
                for family in families
            )
        stored = {row["estimator"]: float(row["strict_macro_top1"])
                  for row in self.csv("final_claim_checks/strict_transfer_metrics.csv")
                  if row["strict_macro_top1"]}
        for model in ("fused_k12", "radio_k12"):
            if abs(result[f"macro_top1_{model}"] - stored[model]) > 1e-12:
                raise ValueError(f"Stored transfer metric differs for {model}")
        return result

    def material(self) -> dict:
        # Per-image holdout predictions are not distributed, so these values
        # are read from the published score tables rather than recomputed.
        holdout = next(row for row in self.csv("final_claim_checks/material_holdout_metrics.csv")
                       if row["estimator"] == "fused_k12" and row["target"] == "six_materials")
        external = self.json("bangladesh_material/roi_metrics.json")
        return {"holdout_n": int(holdout["n_holdout"]),
                "incremental_r2_reported": float(holdout["incremental_r2"]),
                "external_n": int(external["n_primary"]),
                "macro_auroc_reported": float(external["macro_auroc"])}

    def summary(self) -> dict:
        return {"selected_k": self.selected_k(),
                "strict_transfer_recomputed": self.strict_transfer(),
                "material_reported": self.material()}
