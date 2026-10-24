# -*- coding: utf-8 -*-
"""v3_c7a_fetch.py -- C7(a): after the push, fetch every derived-data file cited in MANIFEST anonymously (no token,
no cookies) from raw.githubusercontent.com at the release tag and compare its sha256 with the bytes committed at
that tag in the local repository. Writes submission_package/EJOR/C7A_raw_fetch.json, which v3_package.py uses to
fill the "after push" column of MANIFEST.md. Exit 0 iff 100 % of the files match.

  python code/v3_c7a_fetch.py [--owner lamquocdat01] [--repo awp-frame-scheduling] [--tag v3.0.1-submission]
"""
import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

from v3_io import atomic_write_text  # noqa: E402  (atomic writes, C7)
from v3_package import DATA, PKG, REPO, TAG  # noqa: E402


def git_bytes(rev, path):
    return subprocess.run(["git", "-C", str(REPO), "show", f"{rev}:{path}"], capture_output=True, check=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", default="lamquocdat01")
    ap.add_argument("--repo", default="awp-frame-scheduling")
    ap.add_argument("--tag", default=TAG)
    a = ap.parse_args()
    names = subprocess.run(["git", "-C", str(REPO), "ls-tree", "--name-only", f"{a.tag}:results/awp_cache"],
                           capture_output=True, text=True, check=True).stdout.split()
    paths = DATA + [f"results/awp_cache/{n}" for n in sorted(names)]
    rows, ok = {}, 0
    for p in paths:
        url = f"https://raw.githubusercontent.com/{a.owner}/{a.repo}/{a.tag}/{p}"
        req = urllib.request.Request(url, headers={"User-Agent": "v3_c7a_fetch (anonymous)"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read()
            h = hashlib.sha256(raw).hexdigest()
        except Exception as e:  # noqa: BLE001
            h = f"ERROR {type(e).__name__}: {e}"
        ref = hashlib.sha256(git_bytes(a.tag, p)).hexdigest()
        rows[p] = {"raw_sha256": h, "tag_sha256": ref, "match": h == ref}
        ok += h == ref
        if h != ref:
            print("MISMATCH", p, h)
    out = {"fetched": dt.datetime.now().isoformat(timespec="seconds"), "owner": a.owner, "repo": a.repo, "tag": a.tag,
           "n_files": len(paths), "n_match": ok, "files": rows}
    atomic_write_text(PKG / "C7A_raw_fetch.json", json.dumps(out, indent=1), encoding="utf-8")
    print(f"C7(a): {ok}/{len(paths)} files fetched anonymously at {a.tag} match the committed bytes")
    sys.exit(0 if ok == len(paths) else 1)


if __name__ == "__main__":
    main()
