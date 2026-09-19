from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frontier.phase0h_aggregate import aggregate_curve_points, aggregate_oracle_curve_points, summarize_cells, aggregate_mechanism_metrics
from frontier.utils import load_yaml


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage-dir', default=str(ROOT/'runs'/'phase0h'/'discovery'))
    p.add_argument('--config', default=str(ROOT/'configs'/'phase0h_protocol.yaml'))
    a=p.parse_args()
    d=Path(a.stage_dir)
    protocol=load_yaml(a.config)
    curves=pd.read_csv(d/'curves.csv')
    oracle=pd.read_csv(d/'oracle_curves.csv')
    mech=pd.read_csv(d/'mechanism_metrics.csv')
    cm=aggregate_curve_points(curves)
    om=aggregate_oracle_curve_points(oracle)
    cells=summarize_cells(curves, oracle, float(protocol['primary_targets']['tau_near_optimal_tolerance_accuracy']))
    ms=aggregate_mechanism_metrics(mech)
    cm.to_csv(d/'curve_means.csv',index=False)
    om.to_csv(d/'oracle_curve_means.csv',index=False)
    cells.to_csv(d/'cells.csv',index=False)
    ms.to_csv(d/'mechanism_summary.csv',index=False)
    print(f'cells={len(cells)} curve_means={len(cm)} oracle_curve_means={len(om)}')
if __name__=='__main__': main()
