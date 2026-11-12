"""Convert the Kaggle ASL alphabet images into a CSV of hand-landmark features.

Each row = one image: label, source file, 63 normalised landmark values.
Images in which MediaPipe finds no hand are skipped and counted, giving a
per-class *detection rate* (reported in reports/kaggle_detection_rate.csv).

Usage (from the repo root):
    python -m src.extract_kaggle                     # all images
    python -m src.extract_kaggle --per-class 500     # quick run, 500 per class
    python -m src.extract_kaggle --workers 4
    python -m src.extract_kaggle --classes A B C --append   # process in batches
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import time
from collections import Counter
from multiprocessing import Pool

import cv2

from src.config import load_config
from src.landmarks import HAND_FEATURES, HandTracker, hand_shape_features, primary_hand

MAX_SIDE = 640          # downscale huge images first (MediaPipe resizes internally anyway)
_tracker = None
_pad = 0.0


def _init_worker(model_path, backend, min_conf, pad):
    global _tracker, _pad
    _pad = pad
    _tracker = HandTracker(model_path=model_path, mode="image", num_hands=1, backend=backend,
                           min_detection_confidence=min_conf, min_presence_confidence=min_conf)


def _process(item):
    label, path = item
    img = cv2.imread(path)
    if img is None:
        return label, path, None, "unreadable"
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    if _pad:
        # Many Kaggle crops are so tight that the palm detector misses the hand;
        # a replicated border gives it context (raised detection rate in our tests).
        b = int(max(img.shape[:2]) * _pad)
        img = cv2.copyMakeBorder(img, b, b, b, b, cv2.BORDER_REPLICATE)
    hand = primary_hand(_tracker.process(img))
    if hand is None:
        return label, path, None, "no_hand"
    return label, path, hand_shape_features(hand), "ok"


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kaggle-dir", default=str(cfg["paths"]["kaggle_dir"]))
    ap.add_argument("--out", default=str(cfg["paths"]["alphabet_csv"]))
    ap.add_argument("--per-class", type=int, default=0, help="max images per class (0 = all)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--backend", default="auto", choices=["auto", "tasks", "legacy"])
    ap.add_argument("--classes", nargs="+", help="only these classes (default: all in config)")
    ap.add_argument("--append", action="store_true", help="append to an existing CSV instead of overwriting")
    ap.add_argument("--min-conf", type=float, default=0.3, help="MediaPipe detection confidence")
    ap.add_argument("--pad", type=float, default=0.25, help="border added around each image (fraction)")
    args = ap.parse_args()

    classes = args.classes or cfg["alphabet"]["classes"]
    rng = random.Random(cfg["training"]["seed"])
    items = []
    for label in classes:
        folder = os.path.join(args.kaggle_dir, label)
        if not os.path.isdir(folder):
            print(f"WARNING: missing class folder {folder}")
            continue
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if args.per_class:
            rng.shuffle(files)
            files = files[: args.per_class]
        items += [(label, os.path.join(folder, f)) for f in files]
    print(f"{len(items)} images in {len(classes)} classes, {args.workers} worker(s)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    total, found = Counter(), Counter()
    t0 = time.time()
    append = args.append and os.path.exists(args.out)
    with open(args.out, "a" if append else "w", newline="") as f, \
         Pool(args.workers, initializer=_init_worker,
              initargs=(str(cfg["paths"]["hand_model"]), args.backend, args.min_conf, args.pad)) as pool:
        writer = csv.writer(f)
        if not append:
            writer.writerow(["label", "file"] + [f"f{i}" for i in range(HAND_FEATURES)])
        for i, (label, path, feats, status) in enumerate(pool.imap_unordered(_process, items, chunksize=32), 1):
            total[label] += 1
            if feats is not None:
                found[label] += 1
                writer.writerow([label, os.path.relpath(path, args.kaggle_dir)] + [f"{v:.5f}" for v in feats])
            if i % 1000 == 0 or i == len(items):
                rate = i / (time.time() - t0)
                print(f"  {i}/{len(items)}  ({rate:.1f} img/s, hands found {sum(found.values())})")

    report = cfg["paths"]["reports_dir"] / "kaggle_detection_rate.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    rows = {}
    if append and report.exists():                       # keep results of earlier batches
        with open(report, newline="") as f:
            rows = {r["label"]: r for r in csv.DictReader(f)}
    for label in classes:
        if total[label]:
            rows[label] = {"label": label, "images": total[label], "hands_detected": found[label],
                           "detection_rate": f"{found[label] / total[label]:.3f}"}
    order = cfg["alphabet"]["classes"]
    with open(report, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "images", "hands_detected", "detection_rate"])
        w.writeheader()
        for label in sorted(rows, key=lambda c: order.index(c) if c in order else 99):
            w.writerow(rows[label])
    print(f"\nSaved {sum(found.values())} rows to {args.out}")
    print(f"Overall detection rate: {sum(found.values()) / max(1, sum(total.values())):.1%} "
          f"(per class: {report})")


if __name__ == "__main__":
    main()
