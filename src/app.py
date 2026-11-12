"""Real-time Sign Language Interpreter.

    python -m src.app

Modes (press M to switch):
  SPELL - fingerspelling with the alphabet model (one hand, letter by letter)
  WORD  - whole-word signs with the sequence model (last ~1 s of movement)

Keys:  M mode   SPACE add space   BACKSPACE delete   C clear   S speak   Q quit
Options:  --voice Zira   --rate 2   --list-voices   --camera 1
Fingerspelling: hold a letter steady to type it; drop your hand briefly to
type the same letter twice. The 'space' and 'del' signs work as keys.
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from tensorflow import keras

from src.config import load_config
from src.datasets import add_velocity, canonicalize_sequence
from src.models import compile_predictor
from src.landmarks import HandTracker, draw_hands, frame_features, hand_shape_features, primary_hand


# --------------------------------------------------------------------------- #
class Speaker:
    """Offline text-to-speech on a background thread, so the video never freezes.

    Windows: talks to Microsoft SAPI 5 (Speech API, built into Windows) directly through
             comtypes, keeping one voice object alive. This is the same engine pyttsx3 uses,
             but pyttsx3's event loop on Windows often stays silent after the first sentence.
    Other OS: pyttsx3 (NSSpeechSynthesizer on macOS, eSpeak on Linux), new engine per sentence.
    """

    def __init__(self, voice: str | None = None, rate: int = 0):
        self.q: queue.Queue[str] = queue.Queue()
        self.voice, self.rate = voice, rate
        self.ready = threading.Event()
        self.available = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        if sys.platform == "win32":
            try:
                self._run_sapi()
                return
            except Exception as e:  # noqa: BLE001
                print(f"[speech] SAPI unavailable ({e}); falling back to pyttsx3")
        try:
            self._run_pyttsx3()
        except Exception as e:  # noqa: BLE001
            self.available = False
            self.ready.set()
            print(f"[speech] text-to-speech disabled: {e}")

    def _run_sapi(self):
        import comtypes
        import comtypes.client
        comtypes.CoInitialize()                      # COM must be initialised in this thread
        engine = comtypes.client.CreateObject("SAPI.SpVoice")
        if self.voice:
            voices = engine.GetVoices()
            for i in range(voices.Count):
                token = voices.Item(i)
                if self.voice.lower() in token.GetDescription().lower():
                    engine.Voice = token
                    break
        engine.Rate = self.rate                      # -10 (slow) .. 10 (fast)
        print(f"[speech] Microsoft SAPI 5, voice: {engine.Voice.GetDescription()}")
        self.ready.set()
        while True:
            engine.Speak(self.q.get())               # blocks this thread only, until finished

    def _run_pyttsx3(self):
        import pyttsx3
        print("[speech] pyttsx3")
        self.ready.set()
        while True:
            text = self.q.get()
            engine = pyttsx3.init()
            if self.voice:
                for v in engine.getProperty("voices"):
                    if self.voice.lower() in v.name.lower():
                        engine.setProperty("voice", v.id)
                        break
            engine.say(text)
            engine.runAndWait()
            del engine

    def say(self, text: str) -> bool:
        if not (self.available and text.strip()):
            return False
        self.q.put(text.strip())
        return True


def list_voices():
    """Print the text-to-speech voices installed on this computer."""
    if sys.platform == "win32":
        import comtypes.client
        voices = comtypes.client.CreateObject("SAPI.SpVoice").GetVoices()
        names = [voices.Item(i).GetDescription() for i in range(voices.Count)]
    else:
        import pyttsx3
        names = [v.name for v in pyttsx3.init().getProperty("voices")]
    print("Installed voices:")
    for n in names:
        print("  -", n)


class LetterStabiliser:
    """Accepts a letter only after it has been predicted confidently for N frames in a row."""

    def __init__(self, threshold: float, stable_frames: int):
        self.threshold, self.n = threshold, stable_frames
        self.current, self.count, self.last_committed = None, 0, None

    def reset(self):
        self.current, self.count, self.last_committed = None, 0, None

    def update(self, label: str, conf: float) -> str | None:
        if conf < self.threshold:
            self.current, self.count = None, 0
            return None
        if label == self.current:
            self.count += 1
        else:
            self.current, self.count = label, 1
        if self.count >= self.n and label != self.last_committed:
            self.last_committed = label
            return label
        return None


# --------------------------------------------------------------------------- #
def load(path: Path):
    if not path.exists():
        return None, None
    return keras.models.load_model(path), json.loads(path.with_suffix(".meta.json").read_text())


def overlay(frame, mode, sentence, pred_text, conf, progress, message, detail=""):
    h, w = frame.shape[:2]
    top = 88                                          # header height (two lines)
    panel = frame.copy()
    cv2.rectangle(panel, (0, h - 110), (w, h), (20, 20, 20), -1)
    cv2.rectangle(panel, (0, 0), (w, top), (20, 20, 20), -1)
    frame[:] = cv2.addWeighted(panel, 0.8, frame, 0.2, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    colour = (0, 220, 255) if mode == "SPELL" else (255, 160, 0)
    cv2.putText(frame, f"[{mode}]", (10, 33), font, 0.9, colour, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{pred_text}", (140, 33), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    # confidence bar
    cv2.rectangle(frame, (w - 210, 15), (w - 10, 35), (80, 80, 80), 1)
    cv2.rectangle(frame, (w - 210, 15), (w - 210 + int(200 * conf), 35),
                  (0, 200, 0) if conf >= 0.8 else (0, 140, 255), -1)
    # top-3 candidates, each in its own fixed-width column on the dark header
    for k, (label, p) in enumerate(detail or []):
        x = 10 + k * 160
        text = f"{k + 1}. {label}"
        cv2.putText(frame, text, (x, 70), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        tw = cv2.getTextSize(text, font, 0.6, 1)[0][0]
        cv2.putText(frame, f"{p * 100:.0f}%", (x + tw + 8, 70), font, 0.6, (170, 170, 170), 1, cv2.LINE_AA)
    # stability / window progress
    cv2.rectangle(frame, (0, top), (int(w * progress), top + 4), colour, -1)

    shown = sentence[-45:] + "_"
    cv2.putText(frame, shown, (15, h - 60), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, message or "M mode | SPACE | BKSP delete | C clear | S speak | Q quit",
                (15, h - 20), font, 0.5, (180, 180, 180), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="Real-time sign language interpreter")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--voice", help="part of a voice name, e.g. Zira or David (see --list-voices)")
    ap.add_argument("--rate", type=int, default=0, help="speech speed, -10 (slow) to 10 (fast)")
    ap.add_argument("--list-voices", action="store_true", help="show installed voices and exit")
    args = ap.parse_args()
    if args.list_voices:
        list_voices()
        return
    cfg = load_config()
    inf = cfg["inference"]
    models_dir = cfg["paths"]["models_dir"]
    alpha_model, alpha_meta = load(models_dir / "alphabet_mlp.keras")
    word_path = next((models_dir / f"words_{m}.keras" for m in ("lstm", "gru", "cnn1d")
                      if (models_dir / f"words_{m}.keras").exists()), models_dir / "words_lstm.keras")
    word_model, word_meta = load(word_path)
    alpha_predict = compile_predictor(alpha_model) if alpha_model is not None else None
    word_predict = compile_predictor(word_model) if word_model is not None else None
    if alpha_model is None and word_model is None:
        raise SystemExit("No trained models found in models/. Train one first, e.g.\n"
                         "  python -m src.train --task alphabet --model mlp")

    mp_cfg = cfg["mediapipe"]
    tracker = HandTracker(cfg["paths"]["hand_model"], mode="video", num_hands=mp_cfg["num_hands"],
                          min_detection_confidence=mp_cfg["min_detection_confidence"],
                          min_presence_confidence=mp_cfg["min_presence_confidence"],
                          min_tracking_confidence=mp_cfg["min_tracking_confidence"])
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam.")

    speaker = Speaker(args.voice, args.rate)
    stabiliser = LetterStabiliser(inf["threshold"], inf["stable_frames"])
    T = word_meta["sequence_length"] if word_meta else cfg["words"]["sequence_length"]
    window: deque[np.ndarray] = deque(maxlen=T)
    mode = "SPELL" if alpha_model is not None else "WORD"
    sentence, message, frame_no, cooldown, message_until = "", "", 0, 0, 0
    print(__doc__)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        frame_no += 1
        hands = tracker.process(frame)
        draw_hands(frame, hands)
        pred_text, conf, progress, detail = "-", 0.0, 0.0, []

        if mode == "SPELL":
            hand = primary_hand(hands)
            if hand is None:
                stabiliser.reset()
                pred_text = "no hand"
            else:
                probs = alpha_predict(hand_shape_features(hand)[None])[0]
                i = int(np.argmax(probs))
                label, conf = alpha_meta["labels"][i], float(probs[i])
                top = np.argsort(probs)[::-1][:3]
                detail = [(alpha_meta["labels"][k], float(probs[k])) for k in top]
                pred_text = label
                committed = stabiliser.update(label, conf)
                progress = min(1.0, stabiliser.count / inf["stable_frames"]) if conf >= inf["threshold"] else 0
                if committed == "space":
                    sentence += " "
                elif committed == "del":
                    sentence = sentence[:-1]
                elif committed:
                    sentence += committed
        else:
            window.append(frame_features(hands))
            progress = len(window) / T
            cooldown = max(0, cooldown - 1)
            recent_hands = sum(f[0] + f[67] > 0 for f in list(window)[-10:])
            if len(window) == T and recent_hands >= 5 and frame_no % inf["word_every_n_frames"] == 0:
                x = canonicalize_sequence(np.stack(window))[None]
                if word_meta.get("velocity"):
                    x = add_velocity(x)
                probs = word_predict(x)[0]
                i = int(np.argmax(probs))
                conf = float(probs[i])
                pred_text = word_meta["labels"][i]
                if conf >= inf["threshold"] and cooldown == 0:
                    word = pred_text.replace("iloveyou", "I love you")
                    sentence += ("" if not sentence or sentence.endswith(" ") else " ") + word + " "
                    cooldown = inf["word_cooldown_frames"]
                    window.clear()
            elif not hands:
                pred_text = "no hand"

        if frame_no > message_until:
            message = ""
        overlay(frame, mode, sentence, pred_text, conf, progress, message, detail)
        cv2.imshow("Sign Language Interpreter", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("m"):
            want = "WORD" if mode == "SPELL" else "SPELL"
            if (want == "WORD" and word_model is None) or (want == "SPELL" and alpha_model is None):
                message, message_until = f"No {want.lower()} model trained yet", frame_no + 60
            else:
                mode = want
                window.clear()
                stabiliser.reset()
        elif key == ord(" "):
            sentence += " "
        elif key == 8:                        # backspace
            sentence = sentence[:-1]
        elif key == ord("c"):
            sentence = ""
        elif key == ord("s"):
            if speaker.say(sentence):
                message, message_until = f"Speaking: {sentence.strip()[:40]}", frame_no + 45
            else:
                message, message_until = "Nothing to speak" if speaker.available else "Speech unavailable", frame_no + 45

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
