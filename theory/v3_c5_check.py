# -*- coding: utf-8 -*-
"""v3_c5_check.py -- C5 on the RENDERED PDF: every reference entry must show a DOI, an arXiv id or a URL;
also scans the rendered text for placeholders. Usage: python code/v3_c5_check.py [pdf]"""
import re
import subprocess
import sys
from pathlib import Path

TOPIC = Path(__file__).resolve().parents[1]
pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else TOPIC / "Revision_v3" / "paper_v3.pdf"
txt = pdf.with_suffix(".c5.txt")
subprocess.run(["pdftotext", "-enc", "UTF-8", str(pdf), str(txt)], check=True)
t = txt.read_text(encoding="utf-8", errors="ignore")
ref = t[t.rfind("References"):]
ref = re.sub(r"(?m)^\d+\s*$", "", ref)          # lineno margin numbers on their own line
parts = re.split(r"\n\s*(?:\d+\s+)?\[(\d+)\]\s", "\n" + ref)  # entry starts, with or without a lineno prefix
ents = {int(parts[i]): " ".join(parts[i + 1].split()) for i in range(1, len(parts), 2)}
bad = [k for k, v in ents.items() if not re.search(r"doi:|https?://|arXiv:", v)]
ph = [p for p in ("[TBD", "[TODO", "XXX", "??", "[REPO", "[URL", "[SYSTEM", "[NGÀY", "[DATE") if p in t]
print(f"{pdf.name}: {len(ents)} reference entries; unidentified: {len(bad)} {bad}; placeholders: {ph}")
for k in bad:
    print("  ", k, ents[k][:160])
txt.unlink()
sys.exit(1 if bad or ph else 0)
