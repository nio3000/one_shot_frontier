from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from frontier.phase0r_gates import fit_predictor_with_grouped_cv
from frontier.utils import load_yaml


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "phase0r_protocol.yaml"))
    p.add_argument("--cells", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    protocol = load_yaml(args.config)
    cells = pd.read_csv(args.cells)
    fit = fit_predictor_with_grouped_cv(cells, int(protocol["gates"]["R2"]["grouped_cv_folds"]))
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(fit["cv"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
