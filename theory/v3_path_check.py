# -*- coding: utf-8 -*-
"""v3_path_check.py -- every file path printed in the paper and supplements must exist in the repository.

Scans Revision_v3/{paper_v3,S1_theory_supplement,S2_empirical_supplement}.tex for \\texttt{...} arguments that
look like repository paths (contain "/" or end in .py/.csv/.json/.txt) and checks each with
`git cat-file -e <rev>:<path>` in the repository (AWP_REPO, default D:\\Repos\\awp-frame-scheduling) at the given
revision (default: the tag of v3_package.py if it exists, else HEAD). URLs are listed separately: they point
outside the local repository and cannot be checked at a local tag.

  python code/v3_path_check.py [rev]          (exit 1 if any path is missing)
"""
import os
import re
import subprocess
import sys
from pathlib import Path

TOPIC = Path(__file__).resolve().parents[1]
REV_DIR = TOPIC / "Revision_v3"
REPO = Path(os.environ.get("AWP_REPO", r"D:\Repos\awp-frame-scheduling"))
TEX = ("paper_v3.tex", "S1_theory_supplement.tex", "S2_empirical_supplement.tex")


def git(*a):
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True)


def main():
    rev = sys.argv[1] if len(sys.argv) > 1 else None
    if rev is None:
        from v3_package import TAG
        rev = TAG if git("rev-parse", "-q", "--verify", f"refs/tags/{TAG}").returncode == 0 else "HEAD"
    commit = git("rev-parse", f"{rev}^{{commit}}").stdout.strip()[:7]
    paths, urls = {}, set()
    for f in TEX:
        t = (REV_DIR / f).read_text(encoding="utf-8")
        t = "\n".join(l for l in t.splitlines() if not l.lstrip().startswith("%"))
        for arg in re.findall(r"\\texttt\{([^}]*)\}", t):
            p = arg.replace("\\_", "_")
            if "/" in p or re.search(r"\.(py|csv|json|txt)$", p):
                paths.setdefault(p, set()).add(f)
        urls |= set(re.findall(r"\\url\{([^}]*)\}", t))
    missing = [p for p in sorted(paths) if git("cat-file", "-e", f"{rev}:{p}").returncode != 0]
    print(f"repository {REPO} at {rev} ({commit}): {len(paths)} printed paths, {len(missing)} missing")
    for p in sorted(paths):
        print(f"  {'MISSING' if p in missing else 'ok     '} {p}  ({', '.join(sorted(paths[p]))})")
    print(f"URLs (external, not checkable at a local tag): {sorted(urls)}")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
