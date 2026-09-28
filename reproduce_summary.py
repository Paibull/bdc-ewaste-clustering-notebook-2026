"""Recompute selected paper metrics from the public result tables."""

import json
from pathlib import Path

from reporting.evidence import EvidenceRepository


if __name__ == "__main__":
    evidence = EvidenceRepository(Path(__file__).resolve().parent)
    print(json.dumps(evidence.summary(), indent=2, ensure_ascii=False))
