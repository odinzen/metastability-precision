"""Download the third-party inputs into data/raw, each checked against the checksum of the copy the paper used.

The Materials Project entries (release 2026.09.28) ship with this repository in data/raw/mp and data/raw/mp_r2scan.
Everything else is fetched from its source rather than redistributed:
  - Chen et al. 2024 (doi 10.1038/s44160-024-00502-y): Supplementary Data 2, 7, 8 and 9, and the precursor-selection
    code at commit 6dcee76 (https://github.com/dd-debug/synthesis_planning_algorithm)
  - Szymanski et al. 2023 (doi 10.1038/s41586-023-06734-w): Supplementary Data, the updated file of the corrected
    article, with recipes, outcomes and refinements
  - Wang et al. 2021 (doi 10.1038/s41598-021-94550-5): compiled experimental formation enthalpies (figshare)

Usage: python fetch_inputs.py    (needs git on the path; about 50 MB)
"""
import hashlib
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path

RAW = Path(__file__).resolve().parent / "data" / "raw"
CHEN = "https://static-content.springer.com/esm/art%3A10.1038%2Fs44160-024-00502-y/MediaObjects/44160_2024_502_MOESM{}_ESM.xlsx"
ALAB = "https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-023-06734-w/MediaObjects/41586_2023_6734_MOESM3_ESM.zip"
FILES = [
    (CHEN.format(2), "chen2024/si/MOESM2_ESM.xlsx", "6b6084f75baf127813f1bb654c133dc7"),
    (CHEN.format(7), "chen2024/si/MOESM7_ESM.xlsx", "ab42d8644507077ca01c94e325af7974"),
    (CHEN.format(8), "chen2024/si/MOESM8_ESM.xlsx", "1c9a2197a5b57d7a9774989e421c2091"),
    (CHEN.format(9), "chen2024/si/MOESM9_ESM.xlsx", "76e30081b1e39b8dc7117afdfd6fba17"),
    (ALAB, "alab2023/Supplementary_Data.zip", "d65e1f76c96a8c065c366b18f62723ef"),
    ("https://ndownloader.figshare.com/files/28232538", "measured/expt_formation_enthalpy_kingsbury.json.gz",
     "961033dbfa21626850ee6ee8d8e14d7c"),
]
CHEN_CODE = ("https://github.com/dd-debug/synthesis_planning_algorithm", "6dcee76")


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def fetch(url, rel, checksum):
    out = RAW / rel
    if out.exists() and md5(out) == checksum:
        print(f"have  {rel}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "metastability-precision/1.0"})
    for attempt in range(3):  # figshare occasionally refuses a first request
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                out.write_bytes(r.read())
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(5)
    got = md5(out)
    if got != checksum:
        raise SystemExit(f"{rel}: checksum {got} differs from the paper's copy {checksum}; the source file has changed")
    print(f"got   {rel}")


def main():
    for url, rel, checksum in FILES:
        fetch(url, rel, checksum)
    with zipfile.ZipFile(RAW / "alab2023" / "Supplementary_Data.zip") as z:
        z.extractall(RAW / "alab2023" / "unzipped")
    code = RAW / "chen2024" / "code" / "synthesis_planning_algorithm"
    if not code.exists():
        subprocess.run(["git", "clone", "--quiet", CHEN_CODE[0], str(code)], check=True)
    subprocess.run(["git", "-C", str(code), "checkout", "--quiet", CHEN_CODE[1]], check=True)
    print(f"have  chen2024 code at {CHEN_CODE[1]}")


if __name__ == "__main__":
    main()
