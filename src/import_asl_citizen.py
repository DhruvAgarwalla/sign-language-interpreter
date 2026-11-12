"""Import word signs from Microsoft's ASL Citizen dataset as hand-keypoint sequences.

ASL Citizen: 83,399 webcam videos of 2,731 ASL signs by 52 Deaf / hard-of-hearing
signers, with official train / val / test splits that never share a signer.
https://www.microsoft.com/en-us/research/project/asl-citizen/

The dataset is one 42.8 GB zip. This script does NOT need the whole file: zip
archives keep an index at the end, so with `remotezip` it reads the index over
HTTP and then downloads only the videos of the signs we want (a few hundred MB).
If partial download is not possible, download the zip once and pass --zip PATH.

Each video goes through the SAME MediaPipe pipeline as the live app, idle frames
before/after the sign are trimmed, and the clip is resampled to 30 frames:
    data/words/<sign>/aslc-<participant>_<video>.npy
    data/words/asl_citizen_index.csv   (file, sign, gloss, signer, split)

Usage (from the repo root):
    pip install remotezip
    python -m src.import_asl_citizen --list                    # gloss list -> data/asl_citizen/glosses.csv
    python -m src.import_asl_citizen --find hello thank sorry  # search glosses
    python -m src.import_asl_citizen                           # import the signs in configs/config.yaml
    python -m src.import_asl_citizen --zip D:/ASL_Citizen.zip  # use a downloaded copy instead
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import re
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.config import load_config
from src.datasets import resample_sequence
from src.landmarks import SLOT_FEATURES, HandTracker, frame_features

SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# Archive access (remote or local)
# --------------------------------------------------------------------------- #
def open_archive(source: str):
    if os.path.exists(source):
        import zipfile
        print(f"Opening local zip {source}")
        return zipfile.ZipFile(source)
    try:
        from remotezip import RemoteZip
    except ImportError:
        raise SystemExit("Install remotezip first:  pip install remotezip")
    print("Reading the zip index over HTTP (only the index is downloaded, ~10 MB)...")
    try:
        return RemoteZip(source)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"Could not read the remote zip ({e}).\nThe server may not allow partial downloads. "
            "Download ASL_Citizen.zip once from the project page and run again with --zip PATH.")


def read_splits(zf) -> dict[str, list[dict]]:
    """Returns {split: [ {participant, video, gloss}, ... ]} from splits/*.csv inside the zip."""
    names = zf.namelist()
    out = {}
    for split in SPLITS:
        match = [n for n in names if n.replace("\\", "/").endswith(f"splits/{split}.csv")]
        if not match:
            raise SystemExit(f"splits/{split}.csv not found in the archive")
        text = zf.read(match[0]).decode("utf-8-sig")
        reader = csv.reader(io.StringIO(text))
        next(reader, None)                                   # header
        out[split] = [{"participant": r[0].strip(), "video": r[1].strip(), "gloss": r[2].strip()}
                      for r in reader if len(r) >= 3]
    return out


def video_index(zf) -> dict[str, str]:
    """basename -> path inside the zip, for every video."""
    return {os.path.basename(n): n for n in zf.namelist() if n.lower().endswith((".mp4", ".mov", ".webm"))}


def norm(g: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", g.upper())


# --------------------------------------------------------------------------- #
# Gloss matching
# --------------------------------------------------------------------------- #
def resolve_glosses(signs, aliases, all_glosses) -> dict[str, list[str]]:
    """Map each of our sign names to the ASL Citizen glosses that mean it."""
    by_norm = defaultdict(list)
    for g in all_glosses:
        by_norm[norm(g)].append(g)
        by_norm[re.sub(r"\d+$", "", norm(g))].append(g)        # HELLO2 -> HELLO (variants)
    mapping = {}
    for sign in signs:
        wanted = aliases.get(sign) or [sign]
        found = []
        for w in wanted:
            for g in by_norm.get(norm(w), []):
                if g not in found:
                    found.append(g)
        mapping[sign] = found
    return mapping


# --------------------------------------------------------------------------- #
# Video -> keypoint sequence
# --------------------------------------------------------------------------- #
def extract_sequence(video_bytes: bytes, model_path, backend, suffix=".mp4"):
    """Run MediaPipe over every frame. Returns (T, 134) array or None."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video_bytes)
        path = tmp.name
    try:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        tracker = HandTracker(model_path, mode="video", num_hands=2, backend=backend)
        frames, i = [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame_features(tracker.process(frame, timestamp_ms=int(i * 1000 / fps))))
            i += 1
        cap.release()
        tracker.close()
    finally:
        os.unlink(path)
    return np.stack(frames) if frames else None


def trim_idle(seq: np.ndarray, pad: int = 2, min_frames: int = 8):
    """Keep only the part of the clip where a hand is visible (signers start/end with hands down)."""
    present = (seq[:, 0] + seq[:, SLOT_FEATURES]) > 0
    if present.sum() < min_frames:
        return None
    idx = np.flatnonzero(present)
    return seq[max(0, idx[0] - pad): idx[-1] + pad + 1]


# --------------------------------------------------------------------------- #
def main():
    cfg = load_config()
    ac = cfg.get("asl_citizen", {})
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", default=ac.get("url"), help="local path or URL of ASL_Citizen.zip")
    ap.add_argument("--list", action="store_true", help="save the full gloss list with clip counts and exit")
    ap.add_argument("--find", nargs="+", help="search glosses containing these words and exit")
    ap.add_argument("--signs", nargs="+", default=cfg["words"]["signs"])
    ap.add_argument("--max-per-sign", type=int, default=0, help="limit clips per sign (0 = all)")
    ap.add_argument("--backend", default="auto", choices=["auto", "tasks", "legacy"])
    args = ap.parse_args()

    zf = open_archive(args.zip)
    splits = read_splits(zf)
    counts = {s: Counter(r["gloss"] for r in rows) for s, rows in splits.items()}
    all_glosses = sorted(set().union(*[set(c) for c in counts.values()]))
    print(f"{sum(len(r) for r in splits.values()):,} videos, {len(all_glosses):,} glosses")

    if args.list or args.find:
        out = cfg["paths"]["alphabet_csv"].parent / "asl_citizen" / "glosses.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["gloss", "train", "val", "test"])
            for g in all_glosses:
                w.writerow([g] + [counts[s][g] for s in SPLITS])
        print(f"Gloss list saved to {out}")
        for term in args.find or []:
            hits = [g for g in all_glosses if norm(term) in norm(g)]
            print(f"\n'{term}': " + (", ".join(f"{g} ({sum(counts[s][g] for s in SPLITS)})" for g in hits[:30])
                                    or "no match"))
        return

    mapping = resolve_glosses(args.signs, ac.get("gloss_aliases", {}), all_glosses)
    print("\nSign -> ASL Citizen gloss(es):")
    for sign, gl in mapping.items():
        n = sum(counts[s][g] for s in SPLITS for g in gl)
        print(f"  {sign:10s} -> {', '.join(gl) if gl else 'NOT FOUND (add an alias in config.yaml)'}  [{n} clips]")
    gloss_to_sign = {g: s for s, gl in mapping.items() for g in gl}

    vids = video_index(zf)
    jobs = [(split, r) for split in SPLITS for r in splits[split] if r["gloss"] in gloss_to_sign]
    if args.max_per_sign:
        per, kept = Counter(), []
        for split, r in jobs:
            s = gloss_to_sign[r["gloss"]]
            if per[s] < args.max_per_sign:
                per[s] += 1
                kept.append((split, r))
        jobs = kept
    print(f"\n{len(jobs)} videos to process\n")

    words_dir = cfg["paths"]["words_dir"]
    index_path = words_dir / "asl_citizen_index.csv"
    done = set()
    if index_path.exists():
        with open(index_path, newline="") as f:
            done = {r["video"] for r in csv.DictReader(f)}
    words_dir.mkdir(parents=True, exist_ok=True)
    new_index = not index_path.exists()
    idx_f = open(index_path, "a", newline="")
    idx = csv.writer(idx_f)
    if new_index:
        idx.writerow(["file", "sign", "gloss", "signer", "split", "video", "raw_frames", "kept_frames"])

    T = cfg["words"]["sequence_length"]
    stats, t0, mb = Counter(), time.time(), 0.0
    for k, (split, r) in enumerate(jobs, 1):
        if r["video"] in done:
            stats["already done"] += 1
            continue
        inner = vids.get(os.path.basename(r["video"]))
        if inner is None:
            stats["video missing in zip"] += 1
            continue
        data = zf.read(inner)
        mb += len(data) / 1e6
        seq = extract_sequence(data, cfg["paths"]["hand_model"], args.backend, Path(inner).suffix)
        trimmed = trim_idle(seq) if seq is not None else None
        if trimmed is None:
            stats["no hands found"] += 1
            continue
        sign = gloss_to_sign[r["gloss"]]
        signer = "aslc-" + re.sub(r"[^A-Za-z0-9]", "", r["participant"])
        stem = f"{signer}_{Path(r['video']).stem}"
        (words_dir / sign).mkdir(exist_ok=True)
        np.save(words_dir / sign / f"{stem}.npy", resample_sequence(trimmed, T))
        idx.writerow([stem, sign, r["gloss"], signer, split, r["video"], len(seq), len(trimmed)])
        idx_f.flush()
        stats[f"saved ({split})"] += 1
        if k % 10 == 0 or k == len(jobs):
            el = time.time() - t0
            print(f"  {k}/{len(jobs)}  {mb:.0f} MB downloaded  {el / max(1, k):.1f} s/video")
    idx_f.close()
    print("\nDone:", dict(stats))
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
