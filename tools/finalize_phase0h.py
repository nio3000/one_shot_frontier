from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def read(path): return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None

def main():
    base=ROOT/'runs'/'phase0h'; dg=read(base/'discovery'/'gate_summary.json'); ds=read(base/'discovery'/'sanity_summary.json')
    if dg is None or ds is None: raise RuntimeError('Discovery gate/sanity summary is missing')
    if not ds.get('pass'): decision='HIGH_DIMENSIONAL_TRANSITION_DIRECTION_FALSIFIED'
    elif not dg.get('holdouts_unblocked'):
        decision=dg.get('decision','HIGH_DIMENSIONAL_TRANSITION_DIRECTION_FALSIFIED')
    else:
        ha=read(base/'holdout_a'/'gate_summary.json'); hb=read(base/'holdout_b'/'gate_summary.json')
        if ha is None or hb is None: raise RuntimeError('Discovery passed; both locked holdouts must be completed before finalization')
        pa=bool(ha.get('confirmation',{}).get('pass')); pb=bool(hb.get('confirmation',{}).get('pass'))
        decision='HIGH_DIMENSIONAL_ESTIMABILITY_TRANSITION_SUPPORTED' if pa and pb else 'HIGH_DIMENSIONAL_STRUCTURE_EXISTS_BUT_NOT_TRANSFERABLE'
    out={'final_decision':decision,'discovery':dg,'discovery_sanity_pass':bool(ds.get('pass'))}
    (base/'PHASE0H_FINAL_DECISION.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
