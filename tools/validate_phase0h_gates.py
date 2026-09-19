from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from frontier.phase0h_gates import evaluate_h1,evaluate_h2,evaluate_h3,evaluate_h4,discovery_decision
from frontier.utils import load_yaml

def main():
    p=argparse.ArgumentParser(); p.add_argument('--stage-dir',default=str(ROOT/'runs'/'phase0h'/'discovery')); p.add_argument('--config',default=str(ROOT/'configs'/'phase0h_protocol.yaml')); a=p.parse_args(); d=Path(a.stage_dir)
    protocol=load_yaml(a.config); cells=pd.read_csv(d/'cells.csv'); fit=json.loads((d/'predictor_fit.json').read_text(encoding='utf-8'))
    h1=evaluate_h1(cells,protocol['discovery_gates']['H1']); h2=evaluate_h2(fit,protocol['discovery_gates']['H2']); h3=evaluate_h3(fit,protocol['discovery_gates']['H3']); h4=evaluate_h4(fit,protocol['discovery_gates']['H4']); decision,unblocked=discovery_decision(h1,h2,h3,h4)
    out={'H1':h1,'H2':h2,'H3':h3,'H4':h4,'decision':decision,'holdouts_unblocked':unblocked}; (d/'gate_summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
