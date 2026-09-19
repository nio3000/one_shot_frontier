from pathlib import Path
import importlib.util

def load(rel,name):
    root=Path(__file__).resolve().parents[1]; spec=importlib.util.spec_from_file_location(name,root/rel); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def test_sample_suffix_mapping():
    m=load('tools/fetch_rxrx1_original_sample_channels.py','f')
    assert m._sample_to_channel_suffix('HUVEC-05/Plate4/L07_s2.png',1)=='images/HUVEC-05/Plate4/L07_s2_w1.png'

def test_sample_path_parser():
    m=load('tools/qualify_rxrx1_sample_equivalence.py','q')
    assert m.parse_sample_rel('HUVEC-05/Plate4/L07_s2.png')==('HUVEC-05',4,'L07',2)
