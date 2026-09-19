from pathlib import Path
import importlib.util,pandas as pd
def load():
    p=Path(__file__).resolve().parents[1]/"tools/download_iwildcam_filewise.py"
    s=importlib.util.spec_from_file_location("m",p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def test_bundle():
    assert load().BUNDLE=="0x6313da2b204647e79a14b468131fcd64"
def test_probe_deterministic():
    m=load()
    df=pd.DataFrame({"filename":["c.jpg","a.jpg","b.jpg","d.jpg"],"split":["train","train","test","test"],"location_remapped":[1,1,2,2]})
    assert m.select_probe(df,3)==m.select_probe(df,3)
