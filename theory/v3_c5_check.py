# -*- coding: utf-8 -*-
"""v3_c5_check.py -- C5 on the RENDERED PDF: every reference entry must show a DOI, an arXiv id or a URL;
also scans the rendered text for placeholders. Usage: python code/v3_c5_check.py [pdf]

Two layouts: numbered ([1] ...; elsarticle-num, up to P3C) and author-year (APA via apacite, P4). In author-year
mode the entries are anchored by the first author's surname and the year (keys from the .bbl next to the PDF, or
from Revision_v3/paper_v3.bbl for the packaged copy; author/year from Revision_v3/refs.bib), in list order."""
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

TOPIC = Path(__file__).resolve().parents[1]
REV = TOPIC / "Revision_v3"
pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else REV / "paper_v3.pdf"
txt = pdf.with_suffix(".c5.txt")
subprocess.run(["pdftotext", "-enc", "UTF-8", str(pdf), str(txt)], check=True)
t = txt.read_text(encoding="utf-8", errors="ignore")
txt.unlink()
ref = t[t.rfind("References"):]
ref = re.sub(r"(?m)^\d+\s*$", "", ref)          # page / lineno numbers on their own line
ph = [p for p in ("[TBD", "[TODO", "XXX", "??", "[REPO", "[URL", "[SYSTEM", "[NGÀY", "[DATE") if p in t]
ID = r"doi:|https?://|arXiv"


def fold(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).lower()


if "References" not in t:
    ents = {}
elif re.search(r"\n\s*\[1\]\s", "\n" + ref):
    parts = re.split(r"\n\s*(?:\d+\s+)?\[(\d+)\]\s", "\n" + ref)
    ents = {int(parts[i]): " ".join(parts[i + 1].split()) for i in range(1, len(parts), 2)}
else:
    bbl = pdf.with_suffix(".bbl") if pdf.with_suffix(".bbl").exists() else REV / "paper_v3.bbl"
    keys = re.findall(r"\\APACinsertmetastar\s*\{%?\s*([A-Za-z0-9:_-]+)\}", bbl.read_text(encoding="utf-8"))
    bib = (REV / "refs.bib").read_text(encoding="utf-8")
    flat, ents, pos = fold(" ".join(ref.split())), {}, 0
    starts = []
    for k in keys:
        e = bib[bib.index("{" + k + ","):]
        e = e[: e.index("\n}")]
        au = re.search(r"author\s*=\s*\{(.*?)\},?\n", e, re.S).group(1)
        first = au.split(" and ")[0].strip()
        sur = first.split(",")[0] if "," in first else first.split()[-1]
        sur = fold(re.sub(r"[{}\\'\"`^~]", "", sur)).strip()
        yr = re.search(r"year\s*=\s*\{(\d{4})", e).group(1)
        m = re.compile(r"(?<![a-z])" + re.escape(sur) + r",[^(]{0,400}?\(" + yr).search(flat, pos)
        if not m:
            starts.append((k, None)); continue
        starts.append((k, m.start())); pos = m.start() + 1
    found = [(k, s) for k, s in starts if s is not None]
    for i, (k, s) in enumerate(found):
        end = found[i + 1][1] if i + 1 < len(found) else len(flat)
        ents[k] = flat[s:end]
    for k, s in starts:
        if s is None:
            ents[k] = "<ENTRY NOT FOUND IN PDF>"
bad = [k for k, v in ents.items() if not re.search(ID, v, re.I)]
print(f"{pdf.name}: {len(ents)} reference entries; unidentified: {len(bad)} {bad}; placeholders: {ph}")
for k in bad:
    print("  ", k, ents[k][:160])
sys.exit(1 if bad or ph else 0)
