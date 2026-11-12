"""Dataset loading, train/val/test splitting and landmark augmentation."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.landmarks import HAND_FEATURES, SLOT_FEATURES


# --------------------------------------------------------------------------- #
# Alphabet (static, one frame)
# --------------------------------------------------------------------------- #
def load_alphabet(csv_path: str | Path, classes: list[str]):
    df = pd.read_csv(csv_path)
    df = df[df["label"].isin(classes)]
    X = df[[f"f{i}" for i in range(HAND_FEATURES)]].to_numpy(np.float32)
    y = df["label"].map({c: i for i, c in enumerate(classes)}).to_numpy(np.int64)
    return X, y


def random_split(X, y, val_size=0.15, test_size=0.15, seed=42, groups=None):
    """Stratified train/val/test split."""
    from sklearn.model_selection import train_test_split
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_size / (1 - test_size), stratify=y_tmp, random_state=seed)
    return (X_tr, y_tr), (X_val, y_val), (X_test, y_test)


# --------------------------------------------------------------------------- #
# Words (dynamic, sequences of frames)
# --------------------------------------------------------------------------- #
def canonicalize_sequence(seq: np.ndarray) -> np.ndarray:
    """Put the DOMINANT hand in slot 0, whichever hand it is.

    MediaPipe puts the right hand in slot 0, so a left-handed signer's main hand would land
    in slot 1 and look like a different sign. The dominant hand = the one that is visible
    most and moves most. If it is in slot 1, swap the slots and mirror the wrist x positions
    (shapes are already mirror-normalised), so left- and right-handed signing look the same.
    Used both in training and in the live app.
    """
    S, wx = SLOT_FEATURES, 1 + HAND_FEATURES          # wrist-x offset inside a slot
    def activity(slot):
        b = slot * S
        present = seq[:, b].sum()
        motion = np.abs(np.diff(seq[:, b + wx:b + wx + 2], axis=0)).sum()
        return present + 10 * motion
    if activity(1) <= activity(0):
        return seq
    out = np.concatenate([seq[:, S:2 * S], seq[:, :S]], axis=1).copy()
    out[:, wx] *= -1
    out[:, S + wx] *= -1
    return out


def load_words(words_dir: str | Path, signs: list[str], seq_len: int):
    """Loads data/words/<sign>/<signer>_<n>.npy -> X (N, T, 134), y (N,), signers (N,), stems (N,)"""
    X, y, signers, stems = [], [], [], []
    for idx, sign in enumerate(signs):
        folder = Path(words_dir) / sign
        if not folder.is_dir():
            print(f"WARNING: no recordings for '{sign}' in {folder}")
            continue
        for f in sorted(folder.glob("*.npy")):
            seq = np.load(f).astype(np.float32)
            if seq.shape[0] != seq_len:
                seq = resample_sequence(seq, seq_len)
            X.append(canonicalize_sequence(seq))
            y.append(idx)
            signers.append(f.stem.split("_", 1)[0])   # signer ids never contain "_"
            stems.append(f.stem)
    if not X:
        raise FileNotFoundError(f"No word recordings found in {words_dir}. Run src.collect_data "
                                "or src.import_asl_citizen first.")
    return np.stack(X), np.array(y), np.array(signers), np.array(stems)


def signer_split(X, y, signers, test_signer: str, val_size=0.15, seed=42):
    """Signer-independent split: everything from `test_signer` is the test set."""
    from sklearn.model_selection import train_test_split
    test_mask = signers == test_signer
    if not test_mask.any():
        raise ValueError(f"Signer '{test_signer}' not found. Signers: {sorted(set(signers))}")
    X_rest, y_rest = X[~test_mask], y[~test_mask]
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_rest, y_rest, test_size=val_size, stratify=y_rest, random_state=seed)
    return (X_tr, y_tr), (X_val, y_val), (X[test_mask], y[test_mask])


def resample_sequence(seq: np.ndarray, length: int) -> np.ndarray:
    """Linear interpolation of a (T, F) sequence to `length` frames."""
    t_old = np.linspace(0, 1, len(seq))
    t_new = np.linspace(0, 1, length)
    return np.stack([np.interp(t_new, t_old, seq[:, j]) for j in range(seq.shape[1])], axis=1).astype(np.float32)


# --------------------------------------------------------------------------- #
# Augmentation (operates on normalised landmarks, so it is cheap)
# --------------------------------------------------------------------------- #
def _rotate_xy(pts: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    out = pts.copy()
    out[:, 0] = c * pts[:, 0] - s * pts[:, 1]
    out[:, 1] = s * pts[:, 0] + c * pts[:, 1]
    return out


def mirror_hand(vec: np.ndarray) -> np.ndarray:
    """Left-right mirror of a 63-dim hand (x -> -x)."""
    pts = vec.reshape(21, 3).copy()
    pts[:, 0] *= -1
    return pts.flatten()


def augment_hand(vec: np.ndarray, rng: np.random.Generator,
                 max_rot=np.deg2rad(15), jitter=0.02, mirror_prob=0.5) -> np.ndarray:
    """Random mirror, in-plane rotation, mild axis stretch and Gaussian jitter of one 63-dim hand.

    Mirroring is safe for ASL letters (no two letters are mirror images of each other) and
    makes the model robust to which way the camera / dataset was flipped - e.g. the Kaggle
    'H' points right, but a signer in front of a mirrored webcam points left.
    """
    pts = vec.reshape(21, 3)
    if rng.random() < mirror_prob:
        pts = pts * np.array([-1, 1, 1], dtype=pts.dtype)
    pts = _rotate_xy(pts, rng.uniform(-max_rot, max_rot))
    pts[:, :2] *= rng.uniform(0.9, 1.1, size=2)          # slightly different hand proportions
    pts = pts + rng.normal(0, jitter, pts.shape)
    pts -= pts[0]
    scale = np.max(np.linalg.norm(pts[:, :2], axis=1))
    return (pts / (scale if scale > 1e-6 else 1)).flatten().astype(np.float32)


def augment_alphabet(X, y, copies: int, seed=42, mirror=True):
    """Original data + an exact mirrored copy (if mirror) + `copies` randomly augmented copies."""
    rng = np.random.default_rng(seed)
    Xa = [X]
    if mirror:
        Xa.append(np.stack([mirror_hand(v) for v in X]))
    Xa += [np.stack([augment_hand(v, rng, mirror_prob=0.5 if mirror else 0.0) for v in X])
           for _ in range(copies)]
    return np.concatenate(Xa).astype(np.float32), np.tile(y, len(Xa))


def load_webcam_letters(csv_path: str | Path, classes: list[str]):
    """Webcam letter recordings -> X, y, takes (take 0 = train / calibration, take 1 = test)."""
    df = pd.read_csv(csv_path)
    df = df[df["label"].isin(classes)]
    X = df[[f"f{i}" for i in range(HAND_FEATURES)]].to_numpy(np.float32)
    y = df["label"].map({c: i for i, c in enumerate(classes)}).to_numpy(np.int64)
    return X, y, df["take"].to_numpy(), df["signer"].astype(str).to_numpy()


def augment_sequence(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Time-warp, frame dropout, rotation/jitter of hand shape and position shift."""
    T = len(seq)
    # 1. time warp: speed the sign up / slow it down by up to 20 %
    new_len = int(T * rng.uniform(0.8, 1.2))
    seq = resample_sequence(resample_sequence(seq, max(new_len, 5)), T)
    # 2. frame dropout: repeat the previous frame in place of a few frames
    for t in rng.choice(np.arange(1, T), size=rng.integers(0, 3), replace=False):
        seq[t] = seq[t - 1]
    # 3. mirror the whole clip (camera flipped / video not mirrored): wrist x positions flip
    if rng.random() < 0.5:
        for slot in (0, 1):
            seq[:, slot * SLOT_FEATURES + 1 + HAND_FEATURES] *= -1
    # 4. same rotation for the whole clip, per-frame jitter
    angle = rng.uniform(-np.deg2rad(10), np.deg2rad(10))
    shift = rng.normal(0, 0.03, 2)
    for t in range(T):
        for slot in (0, 1):
            b = slot * SLOT_FEATURES
            if seq[t, b] < 0.5:            # hand not present
                continue
            shape = seq[t, b + 1:b + 1 + HAND_FEATURES].reshape(21, 3)
            shape = _rotate_xy(shape, angle) + rng.normal(0, 0.015, (21, 3))
            seq[t, b + 1:b + 1 + HAND_FEATURES] = shape.flatten()
            seq[t, b + 1 + HAND_FEATURES:b + 3 + HAND_FEATURES] += shift   # wrist x, y
    return seq.astype(np.float32)


def augment_words(X, y, copies: int, seed=42):
    rng = np.random.default_rng(seed)
    Xa = [X] + [np.stack([augment_sequence(s.copy(), rng) for s in X]) for _ in range(copies)]
    return np.concatenate(Xa), np.tile(y, copies + 1)


def add_velocity(X: np.ndarray) -> np.ndarray:
    """Append frame-to-frame differences (motion) to every frame: (N,T,F) -> (N,T,2F)."""
    vel = np.diff(X, axis=1, prepend=X[:, :1])
    return np.concatenate([X, vel], axis=-1)
