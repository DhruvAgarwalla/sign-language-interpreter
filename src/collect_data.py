"""Record word-sign clips from the webcam as landmark sequences.

Each clip = `sequence_length` frames of 134 features, saved to
data/words/<sign>/<signer>_<n>.npy. Only landmarks are stored, never images,
so the dataset is tiny and contains no faces.

Usage:
    python -m src.collect_data --signer dhruv
    python -m src.collect_data --signer priya --signs hello thanks --clips 40

Controls:  SPACE = start / pause    S = skip to next sign    Q = quit
Recording resumes where it stopped (existing clips are counted).
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from src.config import load_config
from src.landmarks import HandTracker, draw_hands, frame_features

BREAK_SECONDS = 1.5


def put(frame, text, y, colour=(255, 255, 255), scale=0.8):
    cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2, cv2.LINE_AA)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signer", required=True, help="your name / id (used for signer-independent tests)")
    ap.add_argument("--signs", nargs="+", default=cfg["words"]["signs"])
    ap.add_argument("--clips", type=int, default=cfg["words"]["clips_per_sign"])
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()
    signer = args.signer.strip().lower().replace("_", "-")
    T = cfg["words"]["sequence_length"]
    mp_cfg = cfg["mediapipe"]

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam. Try --camera 1")

    tracker = HandTracker(cfg["paths"]["hand_model"], mode="video", num_hands=mp_cfg["num_hands"],
                          min_detection_confidence=mp_cfg["min_detection_confidence"],
                          min_presence_confidence=mp_cfg["min_presence_confidence"],
                          min_tracking_confidence=mp_cfg["min_tracking_confidence"])
    quit_all = False
    for sign in args.signs:
        out_dir = cfg["paths"]["words_dir"] / sign
        out_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(out_dir.glob(f"{signer}_*.npy")))
        clip = existing
        recording, paused = False, True
        buffer, break_until = [], 0.0

        while clip < args.clips:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)                       # selfie view
            hands = tracker.process(frame)
            draw_hands(frame, hands)

            now = time.time()
            if not paused and not recording and now >= break_until:
                recording, buffer = True, []
            if recording:
                buffer.append(frame_features(hands))
                cv2.circle(frame, (frame.shape[1] - 30, 30), 12, (0, 0, 255), -1)
                if len(buffer) == T:
                    seq = np.stack(buffer)
                    hand_frames = int((seq[:, 0] + seq[:, 67] > 0).sum())
                    if hand_frames < T // 2:                   # too few frames with a hand: discard
                        print(f"  discarded clip (hand visible in only {hand_frames}/{T} frames)")
                    else:
                        np.save(out_dir / f"{signer}_{clip:03d}.npy", seq)
                        clip += 1
                    recording, break_until = False, now + BREAK_SECONDS

            put(frame, f"Sign: {sign.upper()}   clip {clip}/{args.clips}", 35, (0, 255, 255))
            if paused:
                put(frame, "PAUSED - press SPACE to start", 70, (0, 165, 255))
            elif not recording:
                put(frame, "Get ready...", 70, (0, 255, 0))
            put(frame, "SPACE start/pause   S skip sign   Q quit", frame.shape[0] - 15, scale=0.55)
            cv2.imshow("Collect sign data", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                paused, recording = not paused, False
                break_until = time.time() + BREAK_SECONDS
            elif key == ord("s"):
                break
            elif key == ord("q"):
                quit_all = True
                break
        print(f"{sign}: {clip - existing} new clips ({clip} total for {signer})")
        if quit_all:
            break

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
