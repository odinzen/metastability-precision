"""Write a prediction table with its sha256, without ever overwriting a deposited copy.

A file that already has a .sha256 next to it has been deposited on Zenodo. A rerun then goes to <name>.rerun.csv so
it can be compared with the deposit, and the deposited file and its checksum stay as they are.
"""
import hashlib
from pathlib import Path


def write_hashed(path: Path, header: str, df) -> str:
    if path.with_suffix(path.suffix + ".sha256").exists():
        path = path.with_name(path.stem + ".rerun" + path.suffix)
        print(f"{path.name}: the deposited file is kept; this run is written beside it")
    path.write_text(header + df.to_csv(index=False), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest
