#!/usr/bin/env python3
"""Bin nnUNetv2 validation summary.json by case-name prefix and report per-source dice."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def source_of(case_path: str) -> str:
    return Path(case_path).name.split("_", 1)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary_json", type=Path,
                    help="Path to nnUNet validation summary.json")
    ap.add_argument("--csv", type=Path, default=None,
                    help="Optional CSV dump of per-case results")
    args = ap.parse_args()

    data = json.loads(args.summary_json.read_text())
    per_case = data["metric_per_case"]

    # group: source -> class_id -> [dice, ...]
    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for entry in per_case:
        src = source_of(entry["prediction_file"])
        for cls, m in entry["metrics"].items():
            d = m.get("Dice")
            if d is None or d != d:  # skip NaN (no foreground in GT)
                continue
            groups[src][cls].append(d)

    classes = sorted({c for src in groups.values() for c in src})
    sources = sorted(groups)

    header = ["source", "n"] + [f"L{c}" for c in classes]
    print("\t".join(header))
    for src in sources:
        ns = max(len(v) for v in groups[src].values())
        row = [src, str(ns)]
        for c in classes:
            vals = groups[src][c]
            if not vals:
                row.append("--")
            elif len(vals) == 1:
                row.append(f"{vals[0]:.3f}")
            else:
                row.append(f"{mean(vals):.3f}±{stdev(vals):.3f}")
        print("\t".join(row))

    # global recap (matches summary.json["mean"])
    print()
    print("global mean:", {c: round(mean(d for src in groups.values() for d in src[c]), 4)
                           for c in classes})

    if args.csv:
        import csv
        with args.csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["case", "source"] + [f"L{c}_dice" for c in classes])
            for entry in per_case:
                src = source_of(entry["prediction_file"])
                row = [Path(entry["prediction_file"]).stem, src]
                for c in classes:
                    d = entry["metrics"].get(c, {}).get("Dice")
                    row.append(f"{d:.4f}" if d is not None and d == d else "")
                w.writerow(row)


if __name__ == "__main__":
    main()
