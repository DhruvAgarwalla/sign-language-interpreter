"""Train and evaluate a model.

Examples (run from the repo root):
    # Alphabet from hand landmarks (main model)
    python -m src.train --task alphabet --model mlp
    python -m src.train --task alphabet --model mlp --calibrate   # + your webcam recordings

    # Word signs from landmark sequences
    python -m src.train --task words --model lstm
    python -m src.train --task words --model gru --split signer --test-signer priya
    python -m src.train --task words --model lstm --split aslc            # ASL Citizen official split
    python -m src.train --task words --model lstm --split cross --test-signer dhruv             # ASL Citizen -> your webcam
    python -m src.train --task words --model lstm --split cross --test-signer dhruv --finetune 5

    # Pixel baselines for the comparison experiment
    python -m src.train --task pixels --model smallcnn  --per-class 1500
    python -m src.train --task pixels --model mobilenet --per-class 1500

Outputs: models/<task>_<model>.keras (+ .meta.json with labels/settings),
reports/<experiment>_report.txt, reports/figures/*.png, reports/results.csv
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from src import datasets as ds
from src import models as M
from src.config import load_config
from src.evaluate import evaluate, plot_history


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def callbacks(patience: int):
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, patience // 3)),
    ]


def class_weights(y, n_classes):
    present = np.unique(y)
    w = compute_class_weight("balanced", classes=present, y=y)
    return {int(c): float(v) for c, v in zip(present, w)} | {c: 1.0 for c in range(n_classes) if c not in present}


def fit(model, train, val, cfg, epochs, n_classes):
    (X_tr, y_tr), (X_val, y_val) = train, val
    t0 = time.time()
    hist = model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                     epochs=epochs, batch_size=cfg["training"]["batch_size"],
                     class_weight=class_weights(y_tr, n_classes),
                     callbacks=callbacks(cfg["training"]["patience"]), verbose=2)
    return hist.history, time.time() - t0


def save(model, name: str, cfg, meta: dict):
    out = cfg["paths"]["models_dir"] / f"{name}.keras"
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved model to {out}")


# --------------------------------------------------------------------------- #
def train_alphabet(args, cfg):
    classes = cfg["alphabet"]["classes"]
    X, y = ds.load_alphabet(cfg["paths"]["alphabet_csv"], classes)
    tr, val, te = ds.random_split(X, y, cfg["training"]["val_size"], cfg["training"]["test_size"], args.seed)

    # Optional webcam recordings (src.collect_letters): take 0 = calibration, take 1 = webcam test
    webcam_test = None
    webcam_csv = Path(args.webcam_csv) if args.webcam_csv else cfg["paths"]["alphabet_csv"].parent / "webcam_letters.csv"
    if webcam_csv.exists():
        Xw, yw, takes, _ = ds.load_webcam_letters(webcam_csv, classes)
        if (takes >= 1).any():
            webcam_test = (Xw[takes >= 1], yw[takes >= 1])
        if args.calibrate:
            Xc, yc = Xw[takes == 0], yw[takes == 0]
            tr = (np.concatenate([tr[0]] + [Xc] * args.calib_repeat),
                  np.concatenate([tr[1]] + [yc] * args.calib_repeat))
            print(f"added {len(yc)} webcam calibration samples (x{args.calib_repeat})")
    elif args.calibrate:
        raise SystemExit(f"--calibrate needs {webcam_csv}; record it with src.collect_letters")

    if args.augment:
        tr = ds.augment_alphabet(*tr, copies=cfg["training"]["augment_copies"], seed=args.seed,
                                 mirror=args.mirror)
    print(f"train {len(tr[1])} | val {len(val[1])} | test {len(te[1])} samples"
          + (f" | webcam test {len(webcam_test[1])}" if webcam_test else ""))

    model = M.build_mlp(X.shape[1], len(classes))
    model.summary()
    history, secs = fit(model, tr, val, cfg, args.epochs, len(classes))
    tag = (f"alphabet_{args.model}" + ("" if args.augment else "_noaug")
           + ("" if args.mirror or not args.augment else "_nomirror") + ("_calib" if args.calibrate else ""))
    plot_history(history, cfg["paths"]["reports_dir"] / "figures" / f"{tag}_history.png", tag)
    info = {"input": "landmarks(63)", "augment": args.augment, "train_seconds": round(secs, 1)}
    evaluate(model, *te, classes, cfg["paths"]["reports_dir"], tag, {**info, "split": "kaggle test"})
    # Robustness check: the same test hands, mirrored (simulates a flipped camera / other hand)
    X_mir = np.stack([ds.mirror_hand(v) for v in te[0]])
    evaluate(model, X_mir, te[1], classes, cfg["paths"]["reports_dir"], tag + "_mirroredtest",
             {**info, "split": "kaggle test, mirrored"})
    if webcam_test:
        evaluate(model, *webcam_test, classes, cfg["paths"]["reports_dir"], tag + "_webcamtest",
                 {**info, "split": "webcam test (take 1)"})
    save(model, f"alphabet_{args.model}", cfg, {"task": "alphabet", "labels": classes})


def words_split(args, cfg, X, y, signers, stems):
    """Returns train/val/test according to --split.

    random : stratified random split over all clips
    signer : all clips of --test-signer are the test set
    aslc   : ASL Citizen's official split (train/val/test never share a signer); own clips ignored
    cross  : train on ASL Citizen (train+val+test signers), test on --test-signer's own webcam clips.
             With --finetune N, N clips per sign of that signer are moved into training.
    """
    tr_cfg = cfg["training"]
    if args.split == "random":
        return ds.random_split(X, y, tr_cfg["val_size"], tr_cfg["test_size"], args.seed)
    if args.split == "signer":
        return ds.signer_split(X, y, signers, args.test_signer, tr_cfg["val_size"], args.seed)

    index = cfg["paths"]["words_dir"] / "asl_citizen_index.csv"
    if not index.exists():
        raise SystemExit(f"{index} not found - run src.import_asl_citizen first")
    split_of = dict(pd.read_csv(index)[["file", "split"]].itertuples(index=False, name=None))
    src = np.array([split_of.get(s, "own") for s in stems])
    if args.split == "aslc":
        pick = lambda name: (X[src == name], y[src == name])
        return pick("train"), pick("val"), pick("test")

    # cross-dataset: ASL Citizen -> your webcam
    own = signers == args.test_signer
    if not own.any():
        raise SystemExit(f"No clips from signer '{args.test_signer}'. Record with src.collect_data first.")
    aslc = src != "own"
    tr_mask, val_mask = aslc & (src != "val"), src == "val"
    test_mask = own.copy()
    if args.finetune:
        rng = np.random.default_rng(args.seed)
        for c in np.unique(y[own]):
            ids = np.flatnonzero(own & (y == c))
            moved = rng.choice(ids, size=min(args.finetune, len(ids) - 1), replace=False)
            tr_mask[moved], test_mask[moved] = True, False
    return (X[tr_mask], y[tr_mask]), (X[val_mask], y[val_mask]), (X[test_mask], y[test_mask])


def train_words(args, cfg):
    signs = cfg["words"]["signs"]
    T = cfg["words"]["sequence_length"]
    X, y, signers, stems = ds.load_words(cfg["paths"]["words_dir"], signs, T)
    present = sorted(set(y.tolist()))
    print(f"{len(X)} clips from {len(set(signers))} signers, signs with data: {[signs[i] for i in present]}")

    tr, val, te = words_split(args, cfg, X, y, signers, stems)
    if len(te[1]) == 0 or len(val[1]) == 0:
        raise SystemExit("Empty validation or test set for this split.")
    if args.augment:
        tr = ds.augment_words(*tr, copies=cfg["training"]["augment_copies"], seed=args.seed)
    if args.velocity:
        tr, val, te = [(ds.add_velocity(a), b) for a, b in (tr, val, te)]
    print(f"train {len(tr[1])} | val {len(val[1])} | test {len(te[1])} clips")

    model = M.SEQUENCE_MODELS[args.model](T, tr[0].shape[-1], len(signs))
    model.summary()
    history, secs = fit(model, tr, val, cfg, args.epochs, len(signs))
    split_name = args.split + (f"-{args.test_signer}" if args.split in ("signer", "cross") else "") \
        + (f"-ft{args.finetune}" if args.split == "cross" and args.finetune else "")
    tag = f"words_{args.model}_{split_name}" + ("_vel" if args.velocity else "") + ("" if args.augment else "_noaug")
    plot_history(history, cfg["paths"]["reports_dir"] / "figures" / f"{tag}_history.png", tag)
    evaluate(model, *te, signs, cfg["paths"]["reports_dir"], tag,
             {"input": f"sequence({T}x{tr[0].shape[-1]})", "split": split_name,
              "augment": args.augment, "train_seconds": round(secs, 1)})
    save(model, f"words_{args.model}", cfg,
         {"task": "words", "labels": signs, "sequence_length": T, "velocity": args.velocity})


def train_pixels(args, cfg):
    """Raw-image baselines, streamed from disk with tf.data (no 8 GB arrays in RAM)."""
    classes = cfg["alphabet"]["classes"]
    root = Path(cfg["paths"]["kaggle_dir"])
    rng = random.Random(args.seed)
    paths, labels = [], []
    for i, c in enumerate(classes):
        files = sorted(p for p in (root / c).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        rng.shuffle(files)
        files = files[: args.per_class] if args.per_class else files
        paths += [str(p) for p in files]
        labels += [i] * len(files)
    paths, labels = np.array(paths), np.array(labels)
    (p_tr, y_tr), (p_val, y_val), (p_te, y_te) = ds.random_split(
        paths, labels, cfg["training"]["val_size"], cfg["training"]["test_size"], args.seed)

    size, gray = args.img_size, args.model == "smallcnn"

    def load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, (size, size))
        if gray:
            img = tf.image.rgb_to_grayscale(img)
        return img, label

    def make(p, y, training):
        d = tf.data.Dataset.from_tensor_slices((p, y))
        if training:
            d = d.shuffle(len(p), seed=args.seed)
        d = d.map(load, num_parallel_calls=tf.data.AUTOTUNE)
        if training:
            aug = keras.Sequential([keras.layers.RandomRotation(0.05), keras.layers.RandomZoom(0.1),
                                    keras.layers.RandomBrightness(0.2, value_range=(0, 255))])
            d = d.map(lambda x, l: (aug(x, training=True), l), num_parallel_calls=tf.data.AUTOTUNE)
        return d.batch(32).prefetch(tf.data.AUTOTUNE)

    model = (M.build_smallcnn(size, len(classes)) if gray else M.build_mobilenet(size, len(classes)))
    t0 = time.time()
    hist = model.fit(make(p_tr, y_tr, True), validation_data=make(p_val, y_val, False),
                     epochs=args.epochs, callbacks=callbacks(cfg["training"]["patience"]), verbose=2)
    secs = time.time() - t0
    tag = f"pixels_{args.model}"
    plot_history(hist.history, cfg["paths"]["reports_dir"] / "figures" / f"{tag}_history.png", tag)

    X_te = np.concatenate([x.numpy() for x, _ in make(p_te, y_te, False)])
    evaluate(model, X_te, y_te, classes, cfg["paths"]["reports_dir"], tag,
             {"input": f"image({size}x{size}x{1 if gray else 3})", "split": "random",
              "augment": True, "train_seconds": round(secs, 1)})
    save(model, f"pixels_{args.model}", cfg, {"task": "pixels", "labels": classes, "img_size": size})


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["alphabet", "words", "pixels"])
    ap.add_argument("--model", required=True,
                    choices=["mlp", "lstm", "gru", "cnn1d", "smallcnn", "mobilenet"])
    ap.add_argument("--epochs", type=int, default=cfg["training"]["epochs"])
    ap.add_argument("--no-augment", dest="augment", action="store_false")
    ap.add_argument("--split", default="random", choices=["random", "signer", "aslc", "cross"],
                    help="words: random | signer | aslc (ASL Citizen official) | cross (ASL Citizen -> your webcam)")
    ap.add_argument("--test-signer", help="signer held out for --split signer / cross")
    ap.add_argument("--finetune", type=int, default=0,
                    help="words, --split cross: clips per sign of the test signer added to training")
    ap.add_argument("--velocity", action="store_true", help="words: add frame-to-frame motion features")
    ap.add_argument("--per-class", type=int, default=0, help="pixels: max images per class")
    ap.add_argument("--img-size", type=int, default=0, help="pixels: 64 for smallcnn, 128 for mobilenet")
    ap.add_argument("--no-mirror", dest="mirror", action="store_false", help="alphabet: no mirror augmentation")
    ap.add_argument("--webcam-csv", help="alphabet: webcam recordings (default data/webcam_letters.csv)")
    ap.add_argument("--calibrate", action="store_true", help="alphabet: add webcam take-0 samples to training")
    ap.add_argument("--calib-repeat", type=int, default=5, help="alphabet: oversampling of calibration data")
    ap.add_argument("--seed", type=int, default=cfg["training"]["seed"])
    args = ap.parse_args()
    set_seed(args.seed)

    valid = {"alphabet": {"mlp"}, "words": {"lstm", "gru", "cnn1d"}, "pixels": {"smallcnn", "mobilenet"}}
    if args.model not in valid[args.task]:
        ap.error(f"--task {args.task} supports models: {sorted(valid[args.task])}")
    if args.split in ("signer", "cross") and not args.test_signer:
        ap.error(f"--split {args.split} needs --test-signer NAME")
    if args.task == "pixels" and not args.img_size:
        args.img_size = 64 if args.model == "smallcnn" else 128

    {"alphabet": train_alphabet, "words": train_words, "pixels": train_pixels}[args.task](args, cfg)


if __name__ == "__main__":
    main()
