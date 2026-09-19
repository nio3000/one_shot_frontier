from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from frontier.phase0h_predictors import fit_leave_one_dimension_out

def main():
    p=argparse.ArgumentParser(); p.add_argument('--stage-dir',default=str(ROOT/'runs'/'phase0h'/'discovery')); a=p.parse_args(); d=Path(a.stage_dir)
    cells=pd.read_csv(d/'cells.csv'); curves=pd.read_csv(d/'curve_means.csv')
    fit,pred=fit_leave_one_dimension_out(cells,curves)
    (d/'predictor_fit.json').write_text(json.dumps(fit,indent=2),encoding='utf-8')
    pred.to_csv(d/'predictor_oof_predictions.csv',index=False)
    print(json.dumps({k:{'mae':v['oof_mae'],'r2':v['oof_r2']} for k,v in fit['models'].items()},indent=2))
if __name__=='__main__': main()
