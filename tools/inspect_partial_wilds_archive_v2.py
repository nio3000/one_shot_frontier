from __future__ import annotations
import argparse, gzip, hashlib, json, tarfile
from pathlib import Path, PurePosixPath

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''):
            h.update(b)
    return h.hexdigest()

def _safe_rel(member_name: str) -> Path:
    parts=[p for p in PurePosixPath(member_name.lstrip('./')).parts if p not in ('', '.', '..')]
    return Path(*parts)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--archive', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--max-images', type=int, default=32)
    a=ap.parse_args()
    archive=Path(a.archive); out=Path(a.out_dir)
    meta_dir=out/'metadata'; sample_root=out/'sample_images'
    meta_dir.mkdir(parents=True, exist_ok=True); sample_root.mkdir(parents=True, exist_ok=True)
    members=[]; extracted=[]; image_count=0; truncated=False; err=None
    try:
        with tarfile.open(archive, 'r|gz') as tf:
            for m in tf:
                members.append(m.name)
                if not m.isfile():
                    continue
                low=m.name.lower(); target=None; kind=None
                if low.endswith('metadata.csv') or low.endswith('categories.csv') or 'release_v' in low:
                    target=meta_dir/Path(m.name).name; kind='metadata'
                elif low.endswith(('.png','.jpg','.jpeg')) and image_count < a.max_images:
                    rel=_safe_rel(m.name)
                    if rel.parts and rel.parts[0].lower()=='images': rel=Path(*rel.parts[1:])
                    target=sample_root/rel; target.parent.mkdir(parents=True, exist_ok=True)
                    image_count += 1; kind='sample_image'
                if target is not None:
                    f=tf.extractfile(m)
                    if f is not None:
                        target.write_bytes(f.read())
                        extracted.append({
                            'kind':kind,
                            'member':m.name,
                            'relative_sample_path':str(target.relative_to(sample_root)).replace('\\','/') if kind=='sample_image' else None,
                            'path':str(target),
                            'bytes':target.stat().st_size,
                        })
    except (EOFError, gzip.BadGzipFile, tarfile.ReadError, OSError) as e:
        truncated=True; err=repr(e)
    report={
        'archive':str(archive), 'archive_bytes':archive.stat().st_size, 'archive_sha256':sha256(archive),
        'truncated':truncated, 'error':err, 'members_seen':len(members),
        'first_members':members[:30], 'last_members':members[-30:], 'extracted':extracted,
        'metadata_csv_recovered':any(Path(x['path']).name=='metadata.csv' for x in extracted),
        'sample_images_recovered':sum(x['kind']=='sample_image' for x in extracted), 'sample_paths_preserved':True,
    }
    (out/'partial_archive_recovery_report_v2.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'members_seen_v2.txt').write_text('\n'.join(members),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
