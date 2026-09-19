from pathlib import Path
import importlib.util
import pandas as pd

def load(rel,name):
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location(name,root/rel)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def test_expected_path():
    m=load("tools/plan_rxrx1_segmented_recovery.py","plan")
    row=pd.Series({"experiment":"HUVEC-05","plate":4,"well":"L07","site":2})
    assert m.expected_rel(row)=="images/HUVEC-05/Plate4/L07_s2.png"

def test_bundle_locked():
    m=load("tools/download_rxrx1_incomplete_experiments.py","dl")
    assert m.DEFAULT_BUNDLE=="0x6b7a05a3056a434498f0bb1252eb8440"
