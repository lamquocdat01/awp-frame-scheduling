# -*- coding: utf-8 -*-
"""v3_build_paper.py -- build Revision_v3/paper_v3.tex from paper_v3_template.tex, inserting the
theorem / definition / proof / remark blocks of the TPDS baseline VERBATIM (byte-identical), and
check invariants.

Template markers (alone on a line):   %%BASE{env#k}   -> k-th \\begin{env}...\\end{env} block of the baseline
Usage:
  python code/v3_build_paper.py --list        # list baseline blocks
  python code/v3_build_paper.py               # build paper_v3.tex
  python code/v3_build_paper.py --check       # byte-compare every baseline block with paper_v3.tex
                                              #   -> Revision_v3/INVARIANTS_CHECK_blocks.txt
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path
from v3_io import atomic_write_text  # noqa: E402  (atomic writes, C7)

TOPIC = Path(__file__).resolve().parents[1]
REV = TOPIC / "Revision_v3"
BASE = Path(r"D:\PhD Program\03.Final Submission\03_AwP_Frame_Scheduling\Q1 TPDS\Submission"
            r"\SUBMISSION_PACKAGE_TPDS_20260819\03_LaTeX_Source\paper.tex")
BASE_SHA = "1a7341d4335996b2a30b695833a506590441413d57effb5b327d51dd5d278958"
ENVS = ["lem", "thmone", "thmonep", "thmtwo", "thmthree", "propfour", "propfive", "propsix", "cor",
        "defn", "rem", "proof"]
PAT = re.compile(r"\\begin\{(" + "|".join(ENVS) + r")\}(.*?)\\end\{\1\}", re.S)


def blocks(text: str) -> list[tuple[str, str]]:
    out, cnt = [], {}
    for m in PAT.finditer(text):
        e = m.group(1)
        cnt[e] = cnt.get(e, 0) + 1
        out.append((f"{e}#{cnt[e]}", m.group(0)))
    return out


def base_text() -> str:
    raw = BASE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASE_SHA:
        sys.exit("baseline sha256 mismatch -- refusing to build")
    return raw.decode("utf-8-sig")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    bt = base_text()
    bb = dict(blocks(bt))
    if a.list:
        for k, v in blocks(bt):
            line = bt[: bt.index(v)].count("\n") + 1
            print(f"{k:12s} line {line:4d} len {len(v):5d} :: {v[:90]!r}")
        return
    if a.check:
        paper = (REV / "paper_v3.tex").read_text(encoding="utf-8")
        lines = [f"baseline sha256 {BASE_SHA}", f"paper_v3.tex sha256 {hashlib.sha256(paper.encode()).hexdigest()}", ""]
        n_ok = 0
        for k, v in blocks(bt):
            hit = v in paper
            n_ok += hit
            lines.append(f"{k:12s} sha256 {hashlib.sha256(v.encode()).hexdigest()[:16]}  "
                         f"{'IDENTICAL' if hit else 'NOT PRESENT VERBATIM'}")
        lines.append("")
        lines.append(f"{n_ok}/{len(bb)} baseline blocks present byte-identical in paper_v3.tex")
        atomic_write_text(REV / "INVARIANTS_CHECK_blocks.txt", "\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines[-3:]))
        return
    tpl = (REV / "paper_v3_template.tex").read_text(encoding="utf-8")
    used = []

    def sub(m: re.Match) -> str:
        key = m.group(1)
        if key not in bb:
            sys.exit(f"unknown baseline block {key}")
        used.append(key)
        return bb[key]

    out = re.sub(r"^%%BASE\{([a-z]+#\d+)\}[ \t]*$", sub, tpl, flags=re.M)
    atomic_write_text(REV / "paper_v3.tex", out, encoding="utf-8")
    missing = [k for k in bb if k not in used]
    print(f"paper_v3.tex written; inserted {len(used)} baseline blocks; not inserted: {missing}")


if __name__ == "__main__":
    main()
