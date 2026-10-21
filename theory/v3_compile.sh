#!/usr/bin/env bash
# v3_compile.sh -- rebuild every generated source of AwP v3 and compile paper + supplements + cover letter.
# Usage (from the topic folder):  bash code/v3_compile.sh
set -e
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
python code/v3_tables.py
python code/v3_numbers.py --check
python code/v3_supplement.py
python code/v3_build_paper.py
python code/v3_build_paper.py --check
python code/v3_invariants.py
cd Revision_v3
for f in paper_v3 S1_theory_supplement S2_empirical_supplement cover_letter; do
  pdflatex -interaction=nonstopmode $f.tex >/dev/null 2>&1 || true
  if [ $f = paper_v3 ]; then bibtex paper_v3 >/dev/null 2>&1 || true; fi
  pdflatex -interaction=nonstopmode $f.tex >/dev/null 2>&1 || true
  pdflatex -interaction=nonstopmode $f.tex >/dev/null 2>&1 || true
  echo "$f: $(grep -c '^!' $f.log) errors; $(grep -c 'Overfull' $f.log) overfull; $(grep -o 'Output written.*pages' $f.log)"
done
grep -n "undefined" paper_v3.log | grep -v "Font shape" || echo "no undefined refs/citations"
