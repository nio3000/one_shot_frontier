from pathlib import Path
import importlib.util

def load_tool(rel, name):
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location(name, root/rel)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def test_official_rxrx1_metadata_url_locked():
    m=load_tool("tools/download_rxrx1_original_metadata.py","rxmeta")
    assert m.URL == "https://storage.googleapis.com/rxrx/rxrx1/rxrx1-metadata.zip"
