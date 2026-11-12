# Real-Time ASL Sign Language Interpreter

A webcam application that recognises **American Sign Language (ASL)** and turns it into text and speech.
It has two parts:

| Mode | What it recognises | Model | Input |
|---|---|---|---|
| **SPELL** | Fingerspelling: A–Z, `space`, `del` | MLP | 21 hand keypoints (63 numbers) from one frame |
| **WORD** | Whole-word signs: hello, thanks, yes, no, I love you, please, sorry, help | LSTM / GRU / 1D-CNN | 30-frame sequence of keypoints for both hands |

The models never see raw pixels. **MediaPipe Hand Landmarker** first turns each frame into hand
keypoints, so the models cannot learn the background, lighting or the person's face. They stay small
and fast and need little data.

---

## Architecture

```
Webcam (OpenCV) ─► MediaPipe Hand Landmarker ─► 21 keypoints × (x, y, z) per hand
                                                   │
                       feature engineering: wrist = origin, unit scale,
                       left hands mirrored to right, + wrist position for motion
                                                   │
                 ┌─────────────────────────────────┴───────────────────────────┐
           SPELL: MLP on one frame                          WORD: LSTM on last 30 frames
                 │                                                            │
       confidence ≥ 0.8 and stable 10 frames               confidence ≥ 0.8 + cooldown
                 └──────────────────► sentence builder ─► screen + text-to-speech
```

## Results so far

Alphabet model, trained on hand keypoints extracted from 400 images per class of the Kaggle ASL
alphabet dataset (8,600 images where a hand was detected; 70/15/15 stratified split):

| Experiment | Kaggle test acc. | Macro-F1 | **Mirrored test** macro-F1 | Latency (CPU) |
|---|---|---|---|---|
| Keypoint MLP, no augmentation | 97.1 % | 0.967 | 0.462 | < 1 ms |
| + rotation / jitter augmentation | 98.1 % | 0.979 | 0.487 | < 1 ms |
| **+ mirror augmentation (final)** | **98.0 %** | **0.977** | **0.978** | < 1 ms |
| Prototype 4 CNN (old, pixels) | 93 %* | 0.35 | – | – |

All keypoint models have 54k parameters. The **mirrored test** uses the same test hands flipped left↔right.
It simulates a webcam that mirrors the image, or a left-handed signer. Without mirror augmentation, accuracy
collapses to about 50 %. Live testing showed the same effect: `H` was read as `V` because the Kaggle `H`
always points one way. Mirroring is safe for ASL letters, because no two letters are mirror images of each other.

\* The old model's 93 % came from class imbalance (92 % of test images were "yes"). It scored 0 % on two classes.

Full per-class reports: `reports/*_report.txt`. Plots: `reports/figures/`. All runs are logged in `reports/results.csv`.

**MediaPipe detection rate** on the Kaggle images is 77 % overall (`reports/kaggle_detection_rate.csv`).
Closed-fist letters are hardest (M 56 %, N 43 %), and so are the low-resolution `del`/`space` images.
Adding a 25 % border and lowering detection confidence to 0.3 raised detection from 51 % to 58 %
on a test set of hard classes.

> **Limitation:** the Kaggle split is random, so similar images of the same signer can end up in both
> train and test. Real webcam accuracy on new people will be lower. The word model is evaluated with
> a **signer-independent split** (`--split signer`) for this reason.

---

### Word signs (ASL Citizen, official signer-independent split)

8 signs (hello, thanks, yes, no, I love you, please, sorry, help). Training uses about 14 clips per sign.
The **test set is 97 clips from 11 signers who are not in the training data**.

| Model | Input | Params | Accuracy | Macro-F1 | Latency (CPU) |
|---|---|---|---|---|---|
| LSTM | 30 × 134 keypoints | 159k | 97.9 % | 0.978 | 3.7 ms |
| GRU | 30 × 134 | 122k | 97.9 % | 0.978 | 2.8 ms |
| 1D-CNN | 30 × 134 | 135k | 99.0 % | 0.989 | **1.2 ms** |
| LSTM + velocity features | 30 × 268 | 193k | **99.0 %** | **0.990** | 3.4 ms |

With 97 test clips, one clip is about 1 %, so the four models perform within one or two clips of each other. The
remaining errors are between signs that really are similar: *sorry* → *please* (the same circular movement on the
chest, with a different handshape).

### Cross-dataset: trained on ASL Citizen, tested on our own laptop webcam

The test signer is not in ASL Citizen. Their own clips were recorded with `src/collect_data.py`, 10 per sign.

| Training data | Test clips | Accuracy | Macro-F1 |
|---|---|---|---|
| ASL Citizen only (zero-shot on a new person and camera) | 80 | 95.0 % | 0.947 |
| ASL Citizen + **3 own clips per sign** (fine-tuning) | 56 | **100 %** | **1.000** |

Model: LSTM with velocity features. With no training data from this person or camera, 76 of 80 clips are already correct.
All 4 errors are *yes* clips, misread as *I love you* (3) or *hello* (1). *Yes* is a closed fist, and MediaPipe tracks
closed fists least reliably: the same was seen with the fist letters M, N, A and E, and with the many *yes* clips the
recorder discarded because no hand was detected. Adding just 3 of the signer's own clips per sign fixed every error.

*Caveats:* the test sets are small (on 56 clips, 100 % means the true error rate is very likely below about 5 %). The
fine-tuning and test clips also come from the same recording session (same lighting and background), so the 100 %
is an optimistic estimate.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```
The MediaPipe model (`models/hand_landmarker.task`, ~7 MB) downloads automatically on first run.

## Usage

```bash
# 1. Alphabet: Kaggle images → keypoint CSV → model
python -m src.extract_kaggle --per-class 400        # or omit --per-class for all images
python -m src.train --task alphabet --model mlp

# 2. Words: public data from Microsoft's ASL Citizen + a few of your own clips
python -m src.import_asl_citizen --find hello thank sorry   # check gloss names
python -m src.import_asl_citizen                            # downloads only our signs' videos
python -m src.collect_data --signer dhruv --clips 10        # your webcam test clips
python -m src.train --task words --model lstm --split aslc                                  # official split
python -m src.train --task words --model lstm --split cross --test-signer dhruv             # public -> your webcam
python -m src.train --task words --model lstm --split cross --test-signer dhruv --finetune 3 # + 3 of your clips/sign

# 3. Run the interpreter
python -m src.app

# Optional: calibrate the alphabet to your own webcam/hand
python -m src.collect_letters --signer dhruv            # 2 short takes per letter
python -m src.train --task alphabet --model mlp --calibrate
```
App keys: **M** switch SPELL/WORD · **SPACE** space · **BACKSPACE** delete · **C** clear · **S** speak · **Q** quit

Speech is offline. On Windows it uses **Microsoft SAPI 5**, the speech engine built into Windows, called through
`comtypes`. On macOS/Linux it uses `pyttsx3`. Change the voice with `python -m src.app --list-voices`, then
`--voice Zira --rate 1`.

Pixel baselines for the comparison (`--task pixels --model smallcnn|mobilenet`) stream images from disk with `tf.data`.

## Project structure

```
configs/config.yaml     settings: paths, classes, signs, thresholds
src/landmarks.py        MediaPipe wrapper + feature engineering
src/extract_kaggle.py   Kaggle images → landmark CSV (+ detection-rate report)
src/import_asl_citizen.py  ASL Citizen videos -> keypoint sequences (partial zip download via remotezip)
src/collect_data.py     webcam recorder for word-sign clips (stores keypoints only, no images)
src/collect_letters.py  webcam recorder for letters: calibration data + real webcam test set
src/datasets.py         loading, random / signer-independent splits, keypoint augmentation
src/models.py           MLP, LSTM, GRU, 1D-CNN, pixel baselines
src/train.py            training entry point (one script for every experiment)
src/evaluate.py         accuracy, macro-F1, confusion matrix, latency → reports/results.csv
src/app.py              real-time interpreter with stabilisation and text-to-speech
docs/prototypes.md      history of the earlier prototypes and lessons learned
```

## Roadmap

- [x] Keypoint pipeline + alphabet model
- [x] Mirror augmentation (fixes left/right-pointing letters such as H)
- [ ] Webcam calibration data + webcam test set for the alphabet
- [x] ASL Citizen importer, dominant-hand normalisation, cross-dataset splits
- [x] Import ASL Citizen signs, record own test clips, train LSTM / GRU / 1D-CNN
- [x] Cross-dataset experiment: public only vs + fine-tuning on own clips
- [ ] More test signers (friends) for a stronger signer-independent webcam result
- [ ] Pixel baselines (small CNN, MobileNetV2) for the pixels-vs-keypoints comparison
- [ ] Automatic SPELL/WORD switching based on hand motion
- [ ] TFLite export and a web demo

## Dataset credits
- Word signs: [ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/) (Microsoft Research, 2023):
  83,399 webcam videos, 2,731 signs, 52 Deaf/hard-of-hearing signers. The data is not included in this
  repository; download it with `src/import_asl_citizen.py`. Citation: Desai et al., *ASL Citizen: A
  Community-Sourced Dataset for Advancing Isolated Sign Language Recognition*, NeurIPS 2023 Datasets & Benchmarks.
- ASL alphabet images: Kaggle "American Sign Language Alphabet Dataset" (29 classes)
- Hand landmarks: [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) (Google AI Edge)
