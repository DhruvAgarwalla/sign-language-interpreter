# Earlier prototypes and lessons learned

The current keypoint-based design came after four earlier attempts. Each one taught something that shaped the final system.

## Prototype 1: TensorFlow Object Detection API (SSD MobileNet)
- Followed a public tutorial and collected 76 webcam images (about 15 per sign: hello, thanks, yes, no, iloveyou).
- The images were never annotated with bounding boxes, and the label map still held the tutorial's `Mask/NoMask` classes, so the detector was never trained.
- **Lesson:** object detection needs a lot of labelling work, and a 5-class detector learns *where* a hand is, not the fine finger shape.

## Prototype 2: MS-ASL video dataset (1,000 word classes)
- Parsed the MS-ASL annotations and tried to download and trim the YouTube clips with yt-dlp and FFmpeg.
- Blocked by a path-escaping bug, swapped function arguments, a missing FFmpeg install, and many private or deleted videos. Only 4 clips were obtained.
- **Lesson:** datasets that point to YouTube links decay over time. Self-recorded data, or datasets with downloadable files, are more reliable.

## Prototype 3: Kaggle ASL alphabet (pixels)
- Extracted the dataset (about 87k images, 29 classes) and saved each image as a 64×64 float64 `.npy` file (about 8.5 GB).
- Training never started: loading everything into one array needs more RAM than the laptop has.
- **Lesson:** stream data from disk (`tf.data`), or shrink the representation. Hand keypoints shrink this dataset to a 5 MB CSV.

## Prototype 4: CNN on self-collected photos (5 signs)
- Trained a 3-block CNN (356k params) with augmentation, class weights and early stopping. It reported **93 % accuracy**.
- But the macro-F1 was 0.35, and `hello` and `i_love_you` were never predicted correctly. The reasons:
  1. **Class imbalance:** 3,060 `yes` images against about 50 for each other class. Predicting "yes" every time already scores 92 %.
  2. **Different image sources:** 3,000 of the `yes` images were 200×200 cropped hand photos from a public dataset, while the other classes were full-size phone photos that included the face. The model learned to recognise the image source, not the sign.
  3. **Double normalisation bug:** pixels were divided by 255 twice during training but only once at inference.
  4. **Label-order bug:** training used `os.listdir` order, while the webcam script hard-coded a different order.
  5. **Static frames for moving signs:** hello, thank you, yes and no all involve movement.
- **Lessons:** report macro-F1 and a confusion matrix, not just accuracy. Keep data sources consistent. Save the label list next to the model. Use sequences for moving signs.

## What the final design changes
| Problem | Fix |
|---|---|
| Background, lighting and face leak into the model | MediaPipe keypoints, normalised for position, scale and handedness |
| Too little data for pixels | 63 numbers per hand need far less data |
| Moving signs | 30-frame sequences with an LSTM / GRU / 1D-CNN |
| Misleading accuracy | Macro-F1, confusion matrices, signer-independent test split |
| Label mismatch | Labels stored in `<model>.meta.json` |
| Out of memory | Keypoint CSV; `tf.data` streaming for the pixel baselines |
