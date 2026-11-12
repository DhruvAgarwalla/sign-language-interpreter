"""Record fingerspelling letters from YOUR webcam (calibration + real-world test set).

For every letter you record two short takes (~2 s each):
    take 0 -> added to training ("calibration" data)
    take 1 -> kept aside as a webcam TEST set (the honest real-world accuracy)
Between the two takes, drop your hand and form the letter again so the takes differ.

Rows go to data/webcam_letters.csv (label, signer, take, 63 features). Only
landmarks are stored, never images.

Usage:
    python -m src.collect_letters --signer dhruv                 # all letters
    python -m src.collect_letters --signer dhruv --letters H U V R D

Controls:  SPACE record take   S skip letter   Q quit
Already recorded (letter, signer, take) combinations are skipped, so you can stop and resume.
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import cv2

from src.collect_data import put
from src.config import load_config
from src.landmarks import HAND_FEATURES, HandTracker, draw_hands, hand_shape_features, primary_hand

COUNTDOWN = 1.0


def main():
    cfg = load_config()
    classes = cfg["alphabet"]["classes"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signer", required=True)
    ap.add_argument("--letters", nargs="+", default=classes, help="subset of letters to record")
    ap.add_argument("--frames", type=int, default=60, help="frames (with a hand) per take")
    ap.add_argument("--takes", type=int, default=2)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--out", default=str(cfg["paths"]["alphabet_csv"].parent / "webcam_letters.csv"))
    args = ap.parse_args()
    signer = args.signer.strip().lower()
    letters = [l if l in ("del", "space") else l.upper() for l in args.letters]
    bad = [l for l in letters if l not in classes]
    if bad:
        ap.error(f"unknown letters {bad}; valid: {classes}")

    done = set()
    new_file = not os.path.exists(args.out)
    if not new_file:
        with open(args.out, newline="") as f:
            done = {(r["label"], r["signer"], int(r["take"])) for r in csv.DictReader(f)}

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam. Try --camera 1")
    mp_cfg = cfg["mediapipe"]
    tracker = HandTracker(cfg["paths"]["hand_model"], mode="video", num_hands=1,
                          min_detection_confidence=mp_cfg["min_detection_confidence"],
                          min_presence_confidence=mp_cfg["min_presence_confidence"],
                          min_tracking_confidence=mp_cfg["min_tracking_confidence"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    f = open(args.out, "a", newline="")
    writer = csv.writer(f)
    if new_file:
        writer.writerow(["label", "signer", "take"] + [f"f{i}" for i in range(HAND_FEATURES)])

    quit_all = False
    for letter in letters:
        for take in range(args.takes):
            if (letter, signer, take) in done:
                continue
            state, start, rows = "wait", 0.0, []
            while True:
                ok, frame = cap.read()
                if not ok:
                    quit_all = True
                    break
                frame = cv2.flip(frame, 1)
                hands = tracker.process(frame)
                draw_hands(frame, hands)
                hand = primary_hand(hands)

                if state == "countdown" and time.time() - start >= COUNTDOWN:
                    state = "record"
                if state == "record" and hand is not None:
                    rows.append(hand_shape_features(hand))
                    # small movements while recording make the data more varied - that's good
                    if len(rows) >= args.frames:
                        for v in rows:
                            writer.writerow([letter, signer, take] + [f"{x:.5f}" for x in v])
                        f.flush()
                        print(f"{letter} take {take}: {len(rows)} frames saved")
                        break

                put(frame, f"Letter: {letter}   take {take + 1}/{args.takes}", 35, (0, 255, 255), 1.0)
                if state == "wait":
                    msg = "Form the letter, press SPACE" if take == 0 else "Drop hand, re-form letter, press SPACE"
                    put(frame, msg, 75, (0, 165, 255))
                elif state == "countdown":
                    put(frame, "Get ready...", 75, (0, 255, 0))
                else:
                    put(frame, f"Recording {len(rows)}/{args.frames}  (move slightly)", 75, (0, 0, 255))
                    if hand is None:
                        put(frame, "No hand detected!", 110, (0, 0, 255))
                put(frame, "SPACE record   S skip letter   Q quit", frame.shape[0] - 15, scale=0.55)
                cv2.imshow("Record letters", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(" ") and state == "wait":
                    state, start = "countdown", time.time()
                elif key == ord("s"):
                    break
                elif key == ord("q"):
                    quit_all = True
                    break
            if quit_all or key == ord("s"):
                break
        if quit_all:
            break

    f.close()
    tracker.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
