from __future__ import annotations
import argparse, io, json, os, urllib.request, zipfile
from collections import OrderedDict
from pathlib import Path

RXRX1_IMAGES_ZIP='https://storage.googleapis.com/rxrx/rxrx1/rxrx1-images.zip'

class HTTPRangeReader(io.RawIOBase):
    def __init__(self,url:str,block_size:int=8<<20,max_blocks:int=24):
        self.url=url; self.block_size=int(block_size); self.max_blocks=int(max_blocks); self.pos=0; self.cache=OrderedDict()
        req=urllib.request.Request(url,method='HEAD')
        with urllib.request.urlopen(req,timeout=60) as r:
            self.size=int(r.headers['Content-Length'])
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self,offset,whence=os.SEEK_SET):
        if whence==os.SEEK_SET: p=offset
        elif whence==os.SEEK_CUR: p=self.pos+offset
        elif whence==os.SEEK_END: p=self.size+offset
        else: raise ValueError(whence)
        if p<0: raise ValueError('negative seek')
        self.pos=p; return p
    def _block(self,idx:int)->bytes:
        if idx in self.cache:
            d=self.cache.pop(idx); self.cache[idx]=d; return d
        start=idx*self.block_size; end=min(self.size-1,start+self.block_size-1)
        req=urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{end}'})
        with urllib.request.urlopen(req,timeout=120) as r:
            status=getattr(r,'status',206)
            if status!=206: raise RuntimeError(f'Remote ZIP server did not honor Range: status={status}')
            d=r.read()
        expected=end-start+1
        if len(d)!=expected: raise RuntimeError(f'Range short read {start}-{end}: {len(d)} != {expected}')
        self.cache[idx]=d
        while len(self.cache)>self.max_blocks: self.cache.popitem(last=False)
        return d
    def read(self,size=-1):
        if size is None or size<0: size=self.size-self.pos
        if size<=0 or self.pos>=self.size: return b''
        size=min(size,self.size-self.pos); out=bytearray()
        while size:
            idx=self.pos//self.block_size; off=self.pos%self.block_size; block=self._block(idx)
            n=min(size,len(block)-off); out+=block[off:off+n]; self.pos+=n; size-=n
        return bytes(out)

def _sample_to_channel_suffix(sample_rel:str,channel:int)->str:
    p=Path(sample_rel)
    return (Path('images')/p.parent/f'{p.stem}_w{channel}.png').as_posix()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--recovery-report',required=True); ap.add_argument('--out-dir',required=True)
    ap.add_argument('--url',default=RXRX1_IMAGES_ZIP); ap.add_argument('--max-samples',type=int,default=32); a=ap.parse_args()
    report=json.loads(Path(a.recovery_report).read_text(encoding='utf-8'))
    samples=[x for x in report['extracted'] if x.get('kind')=='sample_image'][:a.max_samples]
    if not samples: raise RuntimeError('No path-preserved WILDS sample images in recovery report')
    desired={}
    for s in samples:
        rel=s['relative_sample_path']
        for ch in (1,2,3): desired[_sample_to_channel_suffix(rel,ch)]=(rel,ch)
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    reader=HTTPRangeReader(a.url)
    print(json.dumps({'remote_zip_bytes':reader.size,'url':a.url,'wanted_members':len(desired)},indent=2))
    found={}
    with zipfile.ZipFile(reader) as zf:
        suffixes=set(desired)
        for name in zf.namelist():
            norm=name.lstrip('./')
            for suffix in suffixes:
                if norm.endswith(suffix): found[suffix]=name
        missing=sorted(set(desired)-set(found))
        if missing: raise RuntimeError(f'Remote original ZIP missing {len(missing)} requested channels; first={missing[:5]}')
        for suffix,name in found.items():
            rel,ch=desired[suffix]; p=Path(rel); target=out/p.parent/f'{p.stem}_w{ch}.png'; target.parent.mkdir(parents=True,exist_ok=True)
            with zf.open(name) as src, target.open('wb') as dst: dst.write(src.read())
    result={'complete':True,'samples':len(samples),'channels_downloaded':len(found),'out_dir':str(out)}
    (out/'selective_download_manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
