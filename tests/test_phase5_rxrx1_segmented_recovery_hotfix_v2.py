from pathlib import Path
import importlib.util
def load():
    p=Path(__file__).resolve().parents[1]/"tools/download_rxrx1_incomplete_experiments.py"
    s=importlib.util.spec_from_file_location("m",p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def test_plate_relative_path():
    m=load()
    assert m._normalized_target_rel("Plate1/A01_s1.png","HEPG2-01").as_posix()=="images/HEPG2-01/Plate1/A01_s1.png"
