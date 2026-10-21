"""
T0 -- Video-level train/test split generator.

Produces:
  splits/S1_fold0.json          -- CDnet2014 within-dataset 60/40 stratified by category
  splits/S2_holdout_<ds>.json   -- cross-dataset leave-one-dataset-out (--scheme S2)

Split unit: VIDEO (not frame, not category).
Stratification: 60/40 within each category independently.
Reproducibility: fixed random seed (default 42).

Usage:
  python scripts/make_splits.py \\
      --cdnet-root "D:/path/to/cdnet2014/dataset" \\
      --seed 42 \\
      --test-size 0.4 \\
      --out-dir splits/

  # Cross-dataset S2:
  python scripts/make_splits.py \\
      --cdnet-root  ... \\
      --bmc-root    ... \\
      --lasiesta-root ... \\
      --virat-root  ... \\
      --scheme S2

JSON schema:
  {
    "scheme": "S1",
    "seed": 42,
    "test_size": 0.4,
    "dataset": "CDnet2014",
    "train": ["category/video", ...],
    "test":  ["category/video", ...],
    "by_category": {
      "<cat>": {"train": [...], "test": [...], "n_train": N, "n_test": N, "flag": ""}
    }
  }
"""
import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def discover_cdnet(dataset_root: str) -> dict[str, list[str]]:
    """Return {category: [video, ...]} for a CDnet-style root."""
    root = Path(dataset_root)
    if not root.exists():
        sys.exit(f"[ERROR] CDnet root not found: {root}")
    by_cat = {}
    for cat_dir in sorted(root.iterdir()):
        if not cat_dir.is_dir():
            continue
        videos = sorted(d.name for d in cat_dir.iterdir() if d.is_dir()
                        and (d / "input").exists())
        if videos:
            by_cat[cat_dir.name] = videos
    return by_cat


def discover_flat(dataset_root: str, dataset_name: str) -> dict[str, list[str]]:
    """
    Generic flat discovery for BMC / LASIESTA / VIRAT.
    Treats the top-level subdirs as 'category=dataset_name', videos=subdir names.
    Adapters may impose their own structure -- this is a best-effort fallback.
    """
    root = Path(dataset_root)
    if not root.exists():
        print(f"[WARN] {dataset_name} root not found: {root} -- skipping")
        return {}
    videos = sorted(d.name for d in root.iterdir() if d.is_dir())
    return {dataset_name: videos} if videos else {}


# ---------------------------------------------------------------------------
# Stratified video-level split
# ---------------------------------------------------------------------------

def stratified_split(
    by_cat: dict[str, list[str]],
    test_size: float,
    seed: int,
    dataset_tag: str = "",
) -> tuple[list[str], list[str], dict]:
    """
    Split at VIDEO level stratified within each category.

    Returns (train_ids, test_ids, by_category_detail)
    video_id format: "{category}/{video}"   (or "{dataset}/{video}" for flat datasets)
    """
    rng = random.Random(seed)

    train_all, test_all = [], []
    by_cat_out = {}

    for cat in sorted(by_cat.keys()):
        videos = sorted(by_cat[cat])
        n = len(videos)

        # Compute test size: ceiling ensures at least 1 in test when n >= 3
        n_test = max(1, math.ceil(n * test_size))
        n_train = n - n_test

        # Shuffle deterministically
        shuffled = videos[:]
        rng.shuffle(shuffled)

        test_vids = shuffled[:n_test]
        train_vids = shuffled[n_test:]

        # video_id = "category/video"
        prefix = f"{cat}" if not dataset_tag else f"{cat}"
        train_ids = [f"{prefix}/{v}" for v in train_vids]
        test_ids = [f"{prefix}/{v}" for v in test_vids]

        flag = ""
        if n_test <= 1:
            flag = "WARNING: n_test<=1"
        if n_train == 0:
            flag = "CRITICAL: n_train=0"

        by_cat_out[cat] = {
            "train": train_ids,
            "test": test_ids,
            "n_train": n_train,
            "n_test": n_test,
            "n_total": n,
            "flag": flag,
        }
        train_all.extend(train_ids)
        test_all.extend(test_ids)

    return train_all, test_all, by_cat_out


# ---------------------------------------------------------------------------
# Integrity check
# ---------------------------------------------------------------------------

def verify_no_overlap(train: list[str], test: list[str]) -> bool:
    overlap = set(train) & set(test)
    if overlap:
        print(f"\n[FAIL] OVERLAP DETECTED ({len(overlap)} videos):")
        for v in sorted(overlap):
            print(f"       {v}")
        return False
    print(f"\n[OK]  train ^ test = empty  (no overlap, {len(train)} train + {len(test)} test = {len(train)+len(test)} total)")
    return True


# ---------------------------------------------------------------------------
# Pretty-print table
# ---------------------------------------------------------------------------

def print_split_table(by_cat: dict, scheme: str, dataset: str):
    width_cat = max(len(c) for c in by_cat) + 2
    sep = "-" * 60
    print(f"\n{sep}")
    print(f"  {scheme}  |  {dataset}")
    print(sep)
    print(f"  {'Category':<{width_cat}}  {'Total':>5}  {'Train':>5}  {'Test':>5}  Flag")
    print(f"  {'-'*width_cat}  {'-----'}  {'-----'}  {'-----'}  {'--------------------'}")
    for cat, info in sorted(by_cat.items()):
        flag = info["flag"]
        marker = "  !! " + flag if flag else ""
        print(f"  {cat:<{width_cat}}  {info['n_total']:>5}  {info['n_train']:>5}  {info['n_test']:>5}{marker}")
    tot = sum(v["n_total"] for v in by_cat.values())
    tr = sum(v["n_train"] for v in by_cat.values())
    te = sum(v["n_test"] for v in by_cat.values())
    print(f"  {'-'*width_cat}  {'-----'}  {'-----'}  {'-----'}")
    print(f"  {'TOTAL':<{width_cat}}  {tot:>5}  {tr:>5}  {te:>5}")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# S2: cross-dataset leave-one-out
# ---------------------------------------------------------------------------

def make_s2_splits(datasets: dict[str, dict[str, list[str]]], test_size: float, seed: int, out_dir: Path):
    """
    For each dataset D: test = all videos of D; train = all videos of other datasets.
    Outputs splits/S2_holdout_<dataset>.json for each D.
    """
    all_video_ids = {}
    for ds_name, by_cat in datasets.items():
        for cat, vids in by_cat.items():
            for v in vids:
                vid_id = f"{cat}/{v}"
                all_video_ids.setdefault(ds_name, []).append(vid_id)

    results = {}
    for holdout_ds in sorted(datasets.keys()):
        holdout_ids = all_video_ids.get(holdout_ds, [])
        train_ids = []
        for ds_name, ids in all_video_ids.items():
            if ds_name != holdout_ds:
                train_ids.extend(ids)

        results[holdout_ds] = {
            "scheme": "S2",
            "holdout_dataset": holdout_ds,
            "seed": seed,
            "train": sorted(train_ids),
            "test": sorted(holdout_ids),
            "n_train": len(train_ids),
            "n_test": len(holdout_ids),
        }

        out_path = out_dir / f"S2_holdout_{holdout_ds}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results[holdout_ds], f, indent=2, ensure_ascii=False)

        ok = verify_no_overlap(train_ids, holdout_ids)
        print(f"  S2 holdout={holdout_ds:12s}  train={len(train_ids):4d}  test={len(holdout_ids):4d}  {'OK' if ok else 'FAIL'}")
        print(f"  Saved → {out_path}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="A3 T0 -- video-level train/test split generator")
    ap.add_argument("--cdnet-root", default="", help="Path to CDnet2014 dataset root (contains category folders)")
    ap.add_argument("--bmc-root",     default="", help="Path to BMC dataset root (for S2)")
    ap.add_argument("--lasiesta-root", default="", help="Path to LASIESTA root (for S2)")
    ap.add_argument("--virat-root",   default="", help="Path to VIRAT root (for S2)")
    ap.add_argument("--seed",      type=int,   default=42,  help="Random seed (default 42)")
    ap.add_argument("--test-size", type=float, default=0.4, help="Fraction of videos in test set (default 0.4)")
    ap.add_argument("--out-dir",   default="splits", help="Output directory (default: splits/)")
    ap.add_argument("--scheme",    choices=["S1", "S2", "both"], default="S1",
                    help="S1=within-dataset, S2=cross-dataset leave-one-out, both=generate all")
    args = ap.parse_args()

    # Resolve CDnet root: fallback to known default if not provided
    cdnet_root = args.cdnet_root or str(
        Path(__file__).resolve().parents[1] / "data" / "cdnet2014" / "dataset"
    )
    # Also try the hardcoded development path as a fallback
    if not Path(cdnet_root).exists():
        dev_fallback = r"D:\THS Programing\06.6 ASMAG Project\dataset cdnet2014\archive\dataset"
        if Path(dev_fallback).exists():
            cdnet_root = dev_fallback
            print(f"[INFO] Using dev CDnet root: {cdnet_root}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    do_s1 = args.scheme in ("S1", "both")
    do_s2 = args.scheme in ("S2", "both")

    # -- S1: CDnet2014 within-dataset split ----------------------------------
    if do_s1:
        print(f"\n{'='*60}")
        print(f"  S1 -- CDnet2014 within-dataset split")
        print(f"  seed={args.seed}  test_size={args.test_size}")
        print(f"{'='*60}")

        by_cat = discover_cdnet(cdnet_root)
        if not by_cat:
            sys.exit("[ERROR] No categories found in CDnet root.")

        train_ids, test_ids, by_cat_detail = stratified_split(
            by_cat, test_size=args.test_size, seed=args.seed, dataset_tag="CDnet2014"
        )

        print_split_table(by_cat_detail, scheme="S1", dataset="CDnet2014")
        ok = verify_no_overlap(train_ids, test_ids)
        if not ok:
            sys.exit("[FATAL] Overlap check failed -- do not use this split.")

        split_doc = {
            "scheme": "S1",
            "fold": 0,
            "seed": args.seed,
            "test_size": args.test_size,
            "dataset": "CDnet2014",
            "dataset_root": cdnet_root,
            "n_train": len(train_ids),
            "n_test": len(test_ids),
            "train": sorted(train_ids),
            "test": sorted(test_ids),
            "by_category": by_cat_detail,
        }
        out_path = out_dir / "S1_fold0.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(split_doc, f, indent=2, ensure_ascii=False)
        print(f"[SAVED] {out_path}")

        # Reproducibility check: re-run with same seed → identical result
        train2, test2, _ = stratified_split(
            by_cat, test_size=args.test_size, seed=args.seed, dataset_tag="CDnet2014"
        )
        assert sorted(train_ids) == sorted(train2) and sorted(test_ids) == sorted(test2), \
            "Reproducibility check FAILED"
        print("[OK]  Reproducibility check passed (re-run gives identical split)")

    # -- S2: cross-dataset leave-one-out ------------------------------------
    if do_s2:
        print(f"\n{'='*60}")
        print(f"  S2 -- cross-dataset leave-one-dataset-out")
        print(f"{'='*60}")
        datasets = {}

        cdnet_by_cat = discover_cdnet(cdnet_root)
        if cdnet_by_cat:
            datasets["CDnet2014"] = cdnet_by_cat

        for ds_name, ds_root in [("BMC", args.bmc_root),
                                  ("LASIESTA", args.lasiesta_root),
                                  ("VIRAT", args.virat_root)]:
            if ds_root:
                flat = discover_flat(ds_root, ds_name)
                if flat:
                    datasets[ds_name] = flat

        if len(datasets) < 2:
            print("[WARN] S2 requires at least 2 datasets. Provide --bmc-root / --lasiesta-root / --virat-root.")
        else:
            make_s2_splits(datasets, test_size=args.test_size, seed=args.seed, out_dir=out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
