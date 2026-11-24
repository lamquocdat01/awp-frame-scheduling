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


# P5B PHA 2 (24-11-2026): EJOR framing. Baseline blocks are inserted through ONE documented transform:
# numbered environments + labels, systems -> OR vocabulary, references to the new labels, and one listed
# MINOR fix (R2 item 30). The --check step compares every baseline block AFTER this transform.
ENV_MAP = {"lem": "lemma", "lemtwo": "lemma", "lemthree": "lemma", "thmone": "proposition", "thmonep": "proposition",
           "thmtwo": "theorem", "thmthree": "theorem", "propfour": "remark", "propfive": "proposition",
           "propsix": "proposition", "cor": "corollary", "defn": "definition", "rem": "remark", "proof": "proof"}
LABELS = {"lem#1": "lem:periodic", "thmtwo#1": "thm:frontier", "defn#1": "def:nonadaptive", "cor#1": "cor:ratio",
          "propfour#1": "rem:smooth", "propsix#1": "prop:tier", "rem#1": "rem:onset"}
# (regex, literal replacement); applied in order
VOCAB = [(r"\\textsc\{SAT-GATE\}", r"\textsc{Sat-Policy}"), (r"HEDGE-GATE", r"\textsc{Hedge}"),
         (r"score-gates", "score policies"), (r"\bgates\b", "policies"), (r"\bgate's\b", "policy's"),
         (r"\bgate\b", "policy"), (r"\bGate\b", "Policy"),
         (r"\bcontains a fire\b", "contains an activation"), (r"\bConsecutive fires\b", "Consecutive activations"),
         (r"\bFires lie\b", "Activations lie"), (r"\bfires\b", "activates"), (r"\bfired\b", "activated"),
         (r"\bfiring\b", "activation"), (r"\bfire\b", "activate"),
         (r"\(top-cue\)", "(top-score)"), (r"\bcues\b", "predictions"), (r"\bcue\b", "prediction"),
         (r"detector hit/miss feedback", "outcome feedback"),
         (r"\bframes-on-event\b", "epochs-on-event"), (r"\bframes\b", "epochs"), (r"\bframe\b", "epoch")]
REFS = [(r"Lemma~1\b", r"Lemma~\ref{lem:periodic}"), (r"Theorem~2\b", r"Theorem~\ref{thm:frontier}"),
        (r"Proposition~6\b", r"Proposition~\ref{prop:tier}"), (r"Def\.~\\ref\{def:sat\}", r"Definition~\ref{def:sat}"),
        (r"\(Assumption~A, Appendix\)", "(Assumption~A below)")]
FIXES = {"thmtwo#1": [(r"\(1\{\+\}\\varepsilon\)-consistent", r"(1{-}\varepsilon)-consistent")],      # R2 item 30
         "rem#4": [(r"two nodes carrying equally many events", "two nodes of equal weight")]}           # R2 item 7


def transform(key: str, block: str) -> str:
    env = key.split("#")[0]
    new = ENV_MAP[env]
    head = re.match(r"\\begin\{" + env + r"\}(\[[^\]]*\])?", block)
    lab = LABELS.get(key)
    out = ("\\begin{" + new + "}" + (head.group(1) or "") + (("\\label{" + lab + "}") if lab else "")
           + block[head.end():])
    out = out[: out.rindex("\\end{" + env + "}")] + "\\end{" + new + "}"
    for pat, rep in VOCAB + REFS + FIXES.get(key, []):
        out = re.sub(pat, lambda _m, r=rep: r, out)
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
        s1 = (REV / "S1_proofs.tex").read_text(encoding="utf-8") if (REV / "S1_proofs.tex").exists() else ""
        lines = [f"baseline sha256 {BASE_SHA}", f"paper_v3.tex sha256 {hashlib.sha256(paper.encode()).hexdigest()}",
                 f"S1_proofs.tex sha256 {hashlib.sha256(s1.encode()).hexdigest()} (complete proofs, supplement S1)", ""]
        n_ok = 0
        for k, v in blocks(bt):
            tv = transform(k, v)
            where = "paper_v3.tex" if tv in paper else ("S1_proofs.tex" if tv in s1 else None)
            n_ok += where is not None
            lines.append(f"{k:12s} sha256 {hashlib.sha256(v.encode()).hexdigest()[:16]} (transformed {hashlib.sha256(tv.encode()).hexdigest()[:16]})  "
                         f"{'IDENTICAL in ' + where if where else 'NOT PRESENT VERBATIM'}")
        lines.append("")
        lines.append(f"{n_ok}/{len(bb)} baseline blocks present byte-identical after the documented transform (paper_v3.tex or S1_proofs.tex)")
        atomic_write_text(REV / "INVARIANTS_CHECK_blocks.txt", "\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines[-3:]))
        return
    used = []

    def sub(m: re.Match) -> str:
        key = m.group(1)
        if key not in bb:
            sys.exit(f"unknown baseline block {key}")
        if key in used:
            sys.exit(f"baseline block {key} inserted twice")
        used.append(key)
        return transform(key, bb[key])

    # P4 (EJOR 30-page limit): long proofs live in supplement S1; S1_proofs_template.tex takes %%BASE markers too
    for src, dst in (("paper_v3_template.tex", "paper_v3.tex"), ("S1_proofs_template.tex", "S1_proofs.tex")):
        if (REV / src).exists():
            out = re.sub(r"^%%BASE\{([a-z]+#\d+)\}[ 	]*$", sub, (REV / src).read_text(encoding="utf-8"), flags=re.M)
            atomic_write_text(REV / dst, out, encoding="utf-8")
    missing = [k for k in bb if k not in used]
    print(f"paper_v3.tex + S1_proofs.tex written; inserted {len(used)} baseline blocks; not inserted: {missing}")


if __name__ == "__main__":
    main()
