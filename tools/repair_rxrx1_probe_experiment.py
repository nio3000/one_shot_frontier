from __future__ import annotations
import argparse,importlib.util,json,shutil,tarfile
from pathlib import Path

def load():
    p=Path(__file__).resolve().parent/"download_rxrx1_incomplete_experiments.py"
    s=importlib.util.spec_from_file_location("dl",p)
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--staging-root",required=True)
    ap.add_argument("--experiment",default="HEPG2-01")
    a=ap.parse_args()
    staging=Path(a.staging_root); exp=a.experiment
    arc=staging/"_segment_downloads"/f"{exp}.tar.gz"
    if not arc.exists() or not tarfile.is_tarfile(arc): raise RuntimeError(f"missing/invalid archive {arc}")
    removed=[]
    for p in sorted((staging/"images").glob("Plate*")):
        if p.is_dir(): removed.append(str(p)); shutil.rmtree(p)
    n=load().extract_experiment_tar(arc,staging,exp)
    r={"experiment":exp,"removed_wrong_roots":removed,"archive_reused":str(arc),"png_written":n}
    print(json.dumps(r,indent=2))
if __name__=="__main__": main()
