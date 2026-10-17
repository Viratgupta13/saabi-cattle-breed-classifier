# 🐄 SAABI: Indian Cattle & Buffalo Breed Classifier

[![CI](https://github.com/Viratgupta13/saabi-cattle-breed-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/Viratgupta13/saabi-cattle-breed-classifier/actions/workflows/ci.yml)

Upload a photo, get the breed. SAABI identifies **38 cattle and buffalo breeds** — Indian indigenous, exotic
dairy and river buffalo — using transfer learning on MobileNetV2, served through a Flask web UI and a JSON API.

> **Test accuracy: 79.9 %** (top-3: **92.7 %**, macro-F1: **0.60**) on 2,636 images held out from training and model selection.
> Also flags uncertain predictions and photos that aren't a supported breed (see *Knowing when it doesn't know*).

**[▶ Live demo](https://saabi-qhel.onrender.com/)** (free tier: the first load after idle can take ~30-60 s to wake up)

![SAABI demo: uploading cattle photos and getting breed predictions](assets/demo.gif)

*Demo: a buffalo and two cattle breeds identified correctly, then a non-cattle photo flagged as unsupported.*

| Confusion matrix (test set) | Training curves |
|---|---|
| ![confusion matrix](assets/confusion_matrix.png) | ![training curves](assets/training_curves.png) |

## Breeds (38)

Alambadi · Amritmahal · Ayrshire · Banni · Bargur · Bhadawari · Brown Swiss · Dangi · Deoni · Gir ·
Guernsey · Hallikar · Hariana · Holstein Friesian · Jaffarabadi · Jersey · Kangayam · Kankrej ·
Kasaragod · Khillari · Krishna Valley · Malnad Gidda · Mehsana · Murrah · Nagori · Nagpuri · Nili Ravi ·
Nimari · Ongole · Pulikulam · Rathi · Red Dane · Red Sindhi · Sahiwal · Tharparkar · Toda ·
Umblachery · Vechur

## Tech stack

| Area | Tools |
|---|---|
| Model | TensorFlow / Keras, MobileNetV2 (ImageNet), scikit-learn, NumPy |
| Inference | TFLite (`ai-edge-litert`), Pillow |
| Serving | Flask, gunicorn |
| DevOps | Docker, GitHub Actions CI, Render |
| Testing | pytest |

## How it works

```
photo ─► augment (train only) ─► scale to [-1,1] ─► MobileNetV2 (ImageNet) ─► GAP ─► Dropout ─► Softmax(38)
```

- **Two-phase transfer learning:** train the classifier head with the backbone frozen, then fine-tune the top 40
  backbone layers at a low learning rate (BatchNorm kept frozen).
- **Mixup augmentation** during fine-tuning (α = 0.2): blends random image pairs and their labels to improve
  generalisation and calibration.
- **Preprocessing lives inside the model.** It takes raw RGB pixels, so training, evaluation and serving share a
  single code path and cannot drift apart (see *Engineering notes*).
- **Imbalance handling:** inverse-frequency class weights during head training.
- **Data pipeline:** breed-name normalisation, MD5 deduplication, corrupt-image filtering, stratified 70/15/15 split.
  Validation drives early stopping and LR scheduling; the test set is touched only by `saabi.evaluate`.

## Results

| Metric (test, n=2,636) | Value |
|---|---|
| Top-1 accuracy | 79.9 % |
| Top-3 accuracy | 92.7 % |
| Macro F1 | 0.60 |

Strongest: Tharparkar (F1 0.92), Vechur (0.92), Sahiwal (0.89), Rathi (0.89), Brown Swiss & Holstein Friesian (0.86).
Weakest: **Krishna Valley** (0.00 — confused with other large white breeds), **Bhadawari** (0.13), **Mehsana** (0.24).
Some visually similar pairs remain challenging: Gir ↔ Sahiwal, Hallikar ↔ Kangayam, Murrah ↔ Nagpuri.
Full per-class numbers are in `models/metrics.json`.

## Knowing when it doesn't know

A classifier that must pick one of 38 breeds will confidently mislabel a dog, a tractor or a landscape. SAABI adds two guards,
both fitted on the **validation split only**:

| Guard | Method | Result |
|---|---|---|
| **Calibration** | Temperature scaling of the softmax (T = 0.97) | Expected calibration error on test: **3.3 % → 2.1 %** |
| **Uncertain flag** | Reject when calibrated confidence is below a validation-chosen threshold | Answers 69 % of test images, with **91.9 %** accuracy on those (vs 79.9 % unfiltered) |
| **"Not a supported breed" flag** | Cosine distance from the image's backbone features to the nearest breed centroid; threshold keeps 95 % of validation images | Filters out-of-distribution images based on feature-space distance to known breed centroids |

The non-cattle benchmark is a **proxy**: 834 random crops of 139 scenery/general photos (macOS wallpaper thumbnails and the
sample photos bundled with scikit-learn/matplotlib). All 834 were flagged. This is an easy case. It was *not* tested on
hard look-alikes (other animals, other cattle breeds), so treat it as a sanity check, not a guarantee.

The first attempt, a max-softmax threshold alone, accepted a temple photo as "Gir" at 83 % confidence, which is why the
feature-space check exists.

## Deployment efficiency (TFLite)

The trained Keras model is exported to TFLite (`python -m saabi.export`) and the web server runs on that, without TensorFlow:

| | Keras / TensorFlow | TFLite |
|---|---|---|
| Model size | 23.5 MB | **9.1 MB** |
| Server RAM (peak) | ~628 MB | **~75 MB** |
| Latency per image (CPU) | 43 ms | **11 ms** |
| Test accuracy | 79.93 % | 79.93 % (100 % prediction agreement, max probability diff 8.1e-6) |

Measured on an Apple-silicon laptop CPU, single image, after warm-up. The smaller footprint is what lets the demo run on a free 512 MB host.

## Engineering notes

- **Found and fixed a train/serve skew bug.** The original scripts trained on OpenCV (BGR) images, but the web
  app fed the model RGB images from PIL. On the same validation set, accuracy was 88.3 % with BGR input but only
  **68.8 %** with the RGB input the app actually sent. Moving normalisation into the model and using one RGB
  loader everywhere removes the failure mode.
- **Corrected an optimistic metric.** The original 88.3 % was measured on the same split used for early stopping and
  on a dataset containing 11 exact duplicates. Re-measured on a clean held-out test split, the honest figure is 83.6 %.
- Server-side validation: file type and size limits, corrupt-image handling, EXIF orientation correction,
  low-confidence warning in the UI.

## Run it

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python app.py                       # http://127.0.0.1:5001
python -m saabi.predict photo.jpg   # CLI top-3
pytest                              # tests
```

API:

```bash
curl -F image=@photo.jpg http://127.0.0.1:5001/api/predict
# {"status":"ok","predictions":[{"breed":"Dangi","confidence":0.998}, ...],"info":{...}}
# status: "ok" | "uncertain" | "unsupported"
```

### Retrain / re-evaluate

Put images in `data/<BreedName>/*.jpg`, then:

```bash
python -m saabi.train
python -m saabi.evaluate   # writes models/metrics.json + assets/confusion_matrix.png
python -m saabi.calibrate  # fits temperature + rejection thresholds (re-run after every retrain)
python -m saabi.export     # exports the TFLite model used by the server and checks parity
```

### Docker

```bash
docker build -t saabi . && docker run -p 10000:10000 saabi   # http://localhost:10000
```

CI (`.github/workflows/ci.yml`) runs the tests and builds and smoke-tests the image on every push. See `DEPLOY.md` for free hosting on Render.

## Project layout

```
saabi/            config, data pipeline, model, train, evaluate, calibrate, export, scoring, predict
app.py            Flask app factory + /api/predict
templates/        web UI
tests/            data-split and API tests
models/           Keras + TFLite models, class names, calibration, metrics
```

## Limitations & next steps

- 17,572 images across 38 breeds; per-class test sets range from ~20–70 images, so per-class numbers still have moderate error bars.
- Visually similar breeds (e.g. Gir ↔ Sahiwal, Hallikar ↔ Kangayam) remain challenging.
- Rejection of non-cattle images was validated on a proxy set only (see above); other animals and unseen breeds are untested.
- Next: Grad-CAM explanations, a stronger backbone with an ablation table, and INT8 quantisation for even smaller models.

## Dataset

Training data is assembled from two public Kaggle datasets via `saabi.merge_datasets`:

| Source | Licence | Images | Breeds |
|---|---|---|---|
| [FarmTrack: Indian Cattle Breed Identification](https://www.kaggle.com/datasets/sujayroy723/farmtrack-indian-cattle-breed-identification) | CC BY 4.0 | ~5,700 | 22 |
| [Indian Cattle Images Dataset](https://www.kaggle.com/datasets/vedanttrivedi/indiancattleimagesdataset) | CC0 / Public Domain | ~14,000 | 32 |

After breed-name normalisation, MD5-based deduplication and pruning breeds with fewer than 70 images,
the merged dataset contains **17,572 images across 38 breeds**. The raw images are not redistributed in this repo.

---

© 2026 Virat Gupta. **All rights reserved.** Code is public for viewing only; reuse needs written permission.
