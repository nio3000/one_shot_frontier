from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

def parse_sample_rel(rel:str):
    p=Path(rel); experiment=p.parts[-3]; plate=int(p.parts[-2].replace('Plate','')); well,site_s=p.stem.rsplit('_s',1)
    return experiment,plate,well,int(site_s)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--recovery-report',required=True); ap.add_argument('--wilds-sample-root',required=True)
    ap.add_argument('--original-channel-root',required=True); ap.add_argument('--original-metadata',required=True)
    ap.add_argument('--out',default='reports/phase5/validation/rxrx1_sample_equivalence.json'); a=ap.parse_args()
    report=json.loads(Path(a.recovery_report).read_text(encoding='utf-8')); samples=[x for x in report['extracted'] if x.get('kind')=='sample_image']
    md=pd.read_csv(a.original_metadata); rows=[]
    for s in samples:
        rel=s['relative_sample_path']; exp,plate,well,site=parse_sample_rel(rel)
        hit=md[(md['experiment'].astype(str)==exp)&(md['plate'].astype(int)==plate)&(md['well'].astype(str)==well)&(md['site'].astype(int)==site)]
        if len(hit)!=1:
            rows.append({'sample':rel,'metadata_match_count':int(len(hit)),'exact_equal':False}); continue
        chans=[]
        for ch in (1,2,3):
            p=Path(a.original_channel_root)/Path(rel).parent/f'{Path(rel).stem}_w{ch}.png'; arr=np.asarray(Image.open(p))
            if arr.ndim!=2: raise ValueError(f'Expected grayscale {p}, got {arr.shape}')
            h,w=arr.shape; y0=(h-256)//2; x0=(w-256)//2; chans.append(arr[y0:y0+256,x0:x0+256])
        recon=np.stack(chans,axis=-1).astype(np.uint8); actual=np.asarray(Image.open(Path(a.wilds_sample_root)/rel).convert('RGB'))
        eq=bool(recon.shape==actual.shape and np.array_equal(recon,actual))
        rows.append({'sample':rel,'metadata_match_count':1,'sirna_id':int(hit.iloc[0]['sirna_id']) if 'sirna_id' in hit.columns else None,
                     'recon_shape':list(recon.shape),'wilds_shape':list(actual.shape),'exact_equal':eq,
                     'max_abs_diff':int(np.max(np.abs(recon.astype(np.int16)-actual.astype(np.int16)))) if recon.shape==actual.shape else None})
    result={'n_samples':len(rows),'metadata_unique_all':all(r['metadata_match_count']==1 for r in rows),
            'pixel_exact_all':bool(rows) and all(r['exact_equal'] for r in rows),
            'max_abs_diff_overall':max((r['max_abs_diff'] for r in rows if r.get('max_abs_diff') is not None),default=None),
            'rows':rows,'promotion_rule':'Exact sample equivalence is necessary but not alone sufficient for full reconstruction promotion.'}
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
