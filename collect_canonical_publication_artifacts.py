from __future__ import annotations
from pathlib import Path
import argparse, hashlib, json, shutil, sys, zipfile
from datetime import datetime, timezone

TEXT_EXTS={'.csv','.json','.yaml','.yml','.md','.txt','.tsv'}
PHASES=('phase1','phase2','phase2r','phase4','phase5')

def sha256(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def load_json(p:Path):
    try: return json.loads(p.read_text(encoding='utf-8'))
    except Exception: return None

def is_formal_complete(obj:dict|None)->bool:
    if not isinstance(obj,dict): return False
    return bool(obj.get('complete')) and bool(obj.get('formal', True)) and str(obj.get('mode','')).lower() not in {'smoke','pair-freeze'}

def select_run_dirs(root:Path):
    selected={k:[] for k in PHASES}
    summaries=[]
    for phase in PHASES:
        base=root/'runs'/phase
        if not base.exists(): continue
        for p in base.rglob('run_summary.json'):
            obj=load_json(p)
            if not is_formal_complete(obj): continue
            # stronger source-lock checks where known
            if phase=='phase1' and not (obj.get('n_conditions')==16 and obj.get('n_curve_rows')==176):
                continue
            if phase=='phase2' and str(obj.get('mode','')).lower() not in {'formal-a','formal'}:
                continue
            if phase=='phase5' and not ((p.parent/'pair_results.csv').exists() and (p.parent/'risk_curves.csv').exists()):
                continue
            selected[phase].append(p.parent)
            summaries.append({'phase':phase,'dir':str(p.parent.relative_to(root)),'run_summary':obj})
    return selected,summaries

def copy_file(src:Path, root:Path, dstroot:Path, records:list):
    rel=src.relative_to(root)
    dst=dstroot/rel
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dst)
    records.append({'path':str(rel).replace('\\','/'),'bytes':src.stat().st_size,'sha256':sha256(src)})

def main():
    ap=argparse.ArgumentParser(description='Collect frozen one_shot_frontier publication artifacts without feature-bank NPZ files.')
    ap.add_argument('--root',default=r'N:\one_shot_frontier')
    ap.add_argument('--out',default='canonical_publication_artifacts.zip')
    args=ap.parse_args()
    root=Path(args.root).resolve()
    if not root.exists(): raise SystemExit(f'ROOT_NOT_FOUND: {root}')
    selected,summaries=select_run_dirs(root)
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    staging=root/f'_publication_artifact_export_{stamp}'
    payload=staging/'canonical_artifacts'
    payload.mkdir(parents=True,exist_ok=False)
    records=[]

    # Frozen configs/manifests/protocols. Deliberately excludes feature NPZ data.
    cfg=root/'configs'
    if cfg.exists():
        for p in cfg.iterdir():
            low=p.name.lower()
            if p.is_file() and p.suffix.lower() in TEXT_EXTS and any(low.startswith(x) for x in PHASES):
                copy_file(p,root,payload,records)

    # All text/tabular artifacts in selected formal-complete run directories.
    for phase,dirs in selected.items():
        for d in dirs:
            for p in d.rglob('*'):
                if p.is_file() and p.suffix.lower() in TEXT_EXTS:
                    copy_file(p,root,payload,records)

    # Phase reports and evidence manifests are useful provenance.
    reports=root/'reports'
    if reports.exists():
        for p in reports.rglob('*'):
            if not p.is_file() or p.suffix.lower() not in TEXT_EXTS: continue
            low=str(p.relative_to(reports)).lower()
            if any(ph in low for ph in PHASES):
                copy_file(p,root,payload,records)

    # Source-lock checks.
    phase5_dirs=selected['phase5']
    checks={
        'phase1_formal_16_conditions_176_rows': bool(selected['phase1']),
        'phase2_formal_selector_run_present': bool(selected['phase2']),
        'phase2r_formal_run_present': bool(selected['phase2r']),
        'phase4_formal_run_present': bool(selected['phase4']),
        'phase5_formal_pair_results_present': any((d/'pair_results.csv').exists() for d in phase5_dirs),
        'phase5_formal_risk_curves_present': any((d/'risk_curves.csv').exists() for d in phase5_dirs),
        'phase5_frozen_pair_manifest_present': (root/'configs'/'phase5_tier1_pair_manifest.json').exists(),
    }
    manifest={
        'export_type':'ONE_SHOT_FRONTIER_CANONICAL_PUBLICATION_ARTIFACTS',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'source_root':str(root),
        'selected_formal_runs':summaries,
        'checks':checks,
        'all_required_groups_found':all(checks.values()),
        'files':sorted(records,key=lambda x:x['path']),
        'note':'Feature-bank NPZ files and source images are intentionally excluded. This export contains frozen tabular/config/provenance artifacts for publication figures and supplements.',
    }
    (payload/'PUBLICATION_ARTIFACT_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    readme=(
        '# Canonical publication artifact export\n\n'
        'Upload this ZIP unchanged. It contains only existing frozen result/config/report files; no experiment is rerun and no values are reconstructed from manuscript prose.\n\n'
        'Required for the current manuscript build:\n'
        '- Phase 1 formal 16-condition / 176-row action curves\n'
        '- Phase 2-A and Phase 2-R formal selector tables (Supplement C)\n'
        '- Phase 4 formal intervention summaries/curves (Figure 2)\n'
        '- Phase 5 frozen pair manifest, pair_results.csv and risk_curves.csv (Figures 3-4, Supplement E)\n'
    )
    (payload/'README_PUBLICATION_EXPORT.md').write_text(readme,encoding='utf-8')

    out=Path(args.out).resolve()
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in payload.rglob('*'):
            if p.is_file(): z.write(p,p.relative_to(staging))
    print(json.dumps({'zip':str(out),'checks':checks,'all_required_groups_found':all(checks.values()),'n_files':len(records)+2},indent=2))
    shutil.rmtree(staging,ignore_errors=True)
    return 0 if all(checks.values()) else 2

if __name__=='__main__':
    raise SystemExit(main())
