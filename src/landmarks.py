"""Hand landmark extraction (MediaPipe) and feature engineering.

Two MediaPipe back-ends are supported:
  * "tasks"  - the current Tasks API (HandLandmarker). Preferred. Needs the
               hand_landmarker.task model file, downloaded automatically.
  * "legacy" - the old mp.solutions.hands API (mediapipe <= 0.10.14). Used only
               as a fallback when the Tasks model cannot be loaded.

Feature vectors
---------------
hand_shape_features(hand)  -> 63 values  (21 landmarks x (x, y, z))
    Translation-invariant (wrist at origin), scale-invariant (max wrist
    distance = 1) and handedness-invariant (left hands are mirrored so every
    hand looks like a right hand). Used by the alphabet model.

frame_features(hands)      -> 134 values (2 hand slots x 67)
    Per slot: 1 presence flag + 63 shape values + wrist (x, y) position and
    hand size in the image. Position and size keep the *movement* information
    that word signs need (e.g. "hello" moves away from the forehead).
    Used by the word (sequence) model.
"""
from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import mediapipe as mp

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")

NUM_LANDMARKS = 21
HAND_FEATURES = NUM_LANDMARKS * 3          # 63
SLOT_FEATURES = 1 + HAND_FEATURES + 3      # 67
FRAME_FEATURES = 2 * SLOT_FEATURES         # 134

# Pairs of landmark indices that form the hand skeleton (for drawing).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                   # palm base
]


@dataclass
class Hand:
    landmarks: np.ndarray   # (21, 3): x, y in [0, 1] image coords, z relative depth
    handedness: str         # "Left" or "Right" (as reported by MediaPipe)
    score: float
    image_size: tuple       # (width, height)


# --------------------------------------------------------------------------- #
# MediaPipe wrapper
# --------------------------------------------------------------------------- #
def ensure_model(path: Path) -> Path:
    """Download the HandLandmarker model if it is not present."""
    path = Path(path)
    if not path.exists() or path.stat().st_size < 1_000_000:      # missing or partial download
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MediaPipe hand model to {path} ...")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(path)
    return path


class HandTracker:
    """Detects up to `num_hands` hands in a BGR frame.

    mode="image" treats every frame independently (datasets).
    mode="video" uses tracking between frames (webcam) - faster and smoother.
    """

    def __init__(self, model_path: Path | None = None, mode: str = "image",
                 num_hands: int = 2, min_detection_confidence: float = 0.5,
                 min_presence_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5, backend: str = "auto"):
        self.mode = mode
        self.backend = None
        self._t0 = time.monotonic()
        self._last_ts = -1

        if backend in ("auto", "tasks"):
            try:
                self._init_tasks(model_path, mode, num_hands, min_detection_confidence,
                                 min_presence_confidence, min_tracking_confidence)
                self.backend = "tasks"
            except Exception as e:  # noqa: BLE001
                if backend == "tasks":
                    raise
                print(f"[HandTracker] Tasks API unavailable ({e}); trying legacy API.")
        if self.backend is None:
            self._init_legacy(mode, num_hands, min_detection_confidence,
                              min_tracking_confidence)
            self.backend = "legacy"

    # -- Tasks API ------------------------------------------------------------
    def _init_tasks(self, model_path, mode, num_hands, det, pres, track):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model_path = ensure_model(model_path or Path("models/hand_landmarker.task"))
        running_mode = vision.RunningMode.VIDEO if mode == "video" else vision.RunningMode.IMAGE
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=running_mode,
            num_hands=num_hands,
            min_hand_detection_confidence=det,
            min_hand_presence_confidence=pres,
            min_tracking_confidence=track,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)

    # -- Legacy API -----------------------------------------------------------
    def _init_legacy(self, mode, num_hands, det, track):
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "Could not load the MediaPipe Tasks model and this mediapipe version has "
                "no legacy API. Download hand_landmarker.task manually from "
                f"{MODEL_URL} into the models/ folder.")
        self._detector = mp.solutions.hands.Hands(
            static_image_mode=(mode != "video"), max_num_hands=num_hands,
            model_complexity=1, min_detection_confidence=det,
            min_tracking_confidence=track)

    # -- Inference -------------------------------------------------------------
    def process(self, frame_bgr: np.ndarray, timestamp_ms: int | None = None) -> list[Hand]:
        """Detect hands. In video mode pass `timestamp_ms` when reading a file (else wall clock)."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        hands: list[Hand] = []

        if self.backend == "tasks":
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            if self.mode == "video":
                ts = timestamp_ms if timestamp_ms is not None else int((time.monotonic() - self._t0) * 1000)
                ts = max(int(ts), self._last_ts + 1)     # timestamps must increase
                self._last_ts = ts
                result = self._detector.detect_for_video(image, ts)
            else:
                result = self._detector.detect(image)
            for lms, cats in zip(result.hand_landmarks, result.handedness):
                arr = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
                hands.append(Hand(arr, cats[0].category_name, float(cats[0].score), (w, h)))
        else:
            result = self._detector.process(rgb)
            if result.multi_hand_landmarks:
                for lms, hd in zip(result.multi_hand_landmarks, result.multi_handedness):
                    arr = np.array([[p.x, p.y, p.z] for p in lms.landmark], dtype=np.float32)
                    c = hd.classification[0]
                    hands.append(Hand(arr, c.label, float(c.score), (w, h)))
        return hands

    def close(self):
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def _to_pixel_space(hand: Hand) -> np.ndarray:
    """Undo MediaPipe's per-axis [0,1] scaling so x and y share one unit."""
    w, h = hand.image_size
    pts = hand.landmarks.astype(np.float32).copy()
    pts[:, 0] *= w
    pts[:, 1] *= h
    pts[:, 2] *= w            # MediaPipe z uses roughly the same scale as x
    return pts


def hand_shape_features(hand: Hand) -> np.ndarray:
    """63-dim translation/scale/handedness-invariant hand shape."""
    pts = _to_pixel_space(hand)
    pts -= pts[0]                                     # wrist -> origin
    if hand.handedness == "Left":
        pts[:, 0] *= -1                               # mirror to canonical right hand
    scale = np.max(np.linalg.norm(pts[:, :2], axis=1))
    pts /= scale if scale > 1e-6 else 1.0
    return pts.flatten().astype(np.float32)


def primary_hand(hands: list[Hand]) -> Hand | None:
    """The most confident / largest hand - used for single-hand fingerspelling."""
    if not hands:
        return None
    def size(hd):
        p = hd.landmarks[:, :2]
        return float(np.ptp(p[:, 0]) * np.ptp(p[:, 1]))
    return max(hands, key=lambda hd: (hd.score, size(hd)))


def frame_features(hands: list[Hand]) -> np.ndarray:
    """134-dim per-frame vector for the word model (slot 0 = Right, slot 1 = Left)."""
    out = np.zeros(FRAME_FEATURES, dtype=np.float32)
    used = set()
    for hand in sorted(hands, key=lambda hd: -hd.score):
        slot = 0 if hand.handedness == "Right" else 1
        if slot in used:                 # two hands with the same label: use the free slot
            slot = 1 - slot
            if slot in used:
                continue
        used.add(slot)
        base = slot * SLOT_FEATURES
        wrist = hand.landmarks[0]
        p = hand.landmarks[:, :2]
        size = float(np.sqrt(np.ptp(p[:, 0]) * np.ptp(p[:, 1])))
        out[base] = 1.0                                               # presence
        out[base + 1: base + 1 + HAND_FEATURES] = hand_shape_features(hand)
        out[base + 1 + HAND_FEATURES:base + SLOT_FEATURES] = [wrist[0] - 0.5, wrist[1] - 0.5, size]
    return out


def draw_hands(frame: np.ndarray, hands: list[Hand]) -> np.ndarray:
    """Draw the hand skeleton on a BGR frame (in place)."""
    h, w = frame.shape[:2]
    for hand in hands:
        pts = [(int(x * w), int(y * h)) for x, y, _ in hand.landmarks]
        colour = (0, 200, 0) if hand.handedness == "Right" else (200, 120, 0)
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], colour, 2)
        for p in pts:
            cv2.circle(frame, p, 3, (255, 255, 255), -1)
    return frame
