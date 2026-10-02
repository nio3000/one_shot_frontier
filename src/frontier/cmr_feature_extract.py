"""Temporary recovery of the original X2 executable; source recovery pending.

The resumed turn accidentally overwrote this untracked source. The original
Python 3.11 bytecode survives and is preserved with its SHA256 in the recovery
record. This shim restores its behavior; it is not the original source and
must not be included in a scientific Git freeze. Replace this file only with
the source whose SHA256 is recorded in the formal X2 evidence manifest.
"""
from pathlib import Path as _RecoveryPath
import hashlib as _recovery_hashlib
import marshal as _recovery_marshal
import sys as _recovery_sys

_recovery_path = (
    _RecoveryPath(__file__).resolve().parents[2]
    / "evidence/cmr_v1/x2/source_recovery/original_cmr_feature_extract.cpython-311.pyc"
)
_recovery_bytes = _recovery_path.read_bytes()
if _recovery_sys.version_info[:2] != (3, 11):
    raise RuntimeError("Original-source recovery pending; preserved bytecode requires Python 3.11")
if _recovery_hashlib.sha256(_recovery_bytes).hexdigest() != "164d597bd096e3643820f68754e93dd8ecc90d4381ad6ed4021c1412c8108505":
    raise RuntimeError("Preserved original X2 bytecode hash mismatch")
exec(_recovery_marshal.loads(_recovery_bytes[16:]), globals())
