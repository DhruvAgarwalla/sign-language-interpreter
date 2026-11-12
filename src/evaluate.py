"""Evaluation: accuracy, macro F1, per-class report, confusion matrix, latency.

Every evaluated model appends one row to reports/results.csv, which becomes
the comparison table in the README / project report.

Standalone usage:
    python -m src.evaluate --model models/alphabet_mlp.keras --task alphabet
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def measure_latency(model, sample: np.ndarray, runs: int = 100) -> float:
    """Mean milliseconds for one forward pass with batch size 1 (real-time setting)."""
    from src.models import compile_predictor
    predict = compile_predictor(model)
    x = sample[None, ...].astype(np.float32)
    for _ in range(10):                       # warm-up (includes graph tracing)
        predict(x)
    t0 = time.perf_counter()
    for _ in range(runs):
        predict(x)
    return (time.perf_counter() - t0) / runs * 1000


def plot_confusion(cm: np.ndarray, class_names: list[str], path: Path, title: str):
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.45), max(5, n * 0.4)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), class_names, rotation=90 if n > 10 else 45)
    ax.set_yticks(range(n), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    if n <= 12:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                        color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(history: dict, path: Path, title: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key in zip(axes, ("accuracy", "loss")):
        ax.plot(history[key], label="train")
        ax.plot(history[f"val_{key}"], label="validation")
        ax.set_xlabel("epoch")
        ax.set_title(key)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate(model, X_test, y_test, class_names: list[str], reports_dir: Path, tag: str,
             extra: dict | None = None) -> dict:
    reports_dir = Path(reports_dir)
    (reports_dir / "figures").mkdir(parents=True, exist_ok=True)

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    labels = list(range(len(class_names)))
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", labels=labels, zero_division=0)
    report = classification_report(y_test, y_pred, labels=labels, target_names=class_names,
                                   zero_division=0, digits=3)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    latency = measure_latency(model, X_test[0])

    (reports_dir / f"{tag}_report.txt").write_text(
        f"{tag}\naccuracy={acc:.4f}  macro_f1={macro_f1:.4f}  latency_ms={latency:.2f}\n\n{report}")
    plot_confusion(cm, class_names, reports_dir / "figures" / f"{tag}_confusion.png",
                   f"{tag}  (acc {acc:.3f}, macro-F1 {macro_f1:.3f})")

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment": tag,
        "model": model.name,
        "params": model.count_params(),
        "test_samples": len(y_test),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "latency_ms": round(latency, 2),
        **(extra or {}),
    }
    results = reports_dir / "results.csv"
    new = not results.exists()
    with open(results, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)

    print(f"\n=== {tag} ===\naccuracy {acc:.4f} | macro-F1 {macro_f1:.4f} | "
          f"{latency:.2f} ms/sample | {model.count_params():,} params\n{report}")
    return row


def main():
    from tensorflow import keras
    from src.config import load_config
    from src.datasets import load_alphabet, random_split

    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", default="alphabet", choices=["alphabet"])
    args = ap.parse_args()

    model = keras.models.load_model(args.model)
    labels = json.loads(Path(args.model).with_suffix(".meta.json").read_text())["labels"]
    X, y = load_alphabet(cfg["paths"]["alphabet_csv"], labels)
    _, _, (X_te, y_te) = random_split(X, y, cfg["training"]["val_size"],
                                      cfg["training"]["test_size"], cfg["training"]["seed"])
    evaluate(model, X_te, y_te, labels, cfg["paths"]["reports_dir"], Path(args.model).stem + "_reeval")


if __name__ == "__main__":
    main()
