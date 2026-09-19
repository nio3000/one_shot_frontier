
from pathlib import Path
import importlib.util

def load(rel,name):
    r=Path(__file__).resolve().parents[1]
    s=importlib.util.spec_from_file_location(name,r/rel)
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def test_parse_member():
    m=load("tools/select_extract_rxrx1_stratified_wilds_samples.py","sel")
    x=m.parse_member("./images/HUVEC-05/Plate4/L07_s2.png")
    assert x["experiment"]=="HUVEC-05" and x["plate"]==4 and x["site"]==2 and x["well"]=="L07"

def test_expected_metadata_constants():
    m=load("tools/audit_rxrx1_original_metadata.py","aud")
    assert m.EXPECTED["n_rows"]==125510
    assert m.EXPECTED["n_experiments"]==51
    assert m.EXPECTED["n_classes"]==1139
