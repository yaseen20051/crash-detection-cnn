# 🚧 CCTV Accident Detection — EfficientNetB0 Transfer Learning

A binary image classifier that detects traffic accidents in CCTV footage frames, built on top of **EfficientNetB0** with a lightweight, regularized classification head and a two-stage (frozen → fine-tuned) transfer learning strategy.

The model reaches **92% test accuracy** and a **0.955 ROC-AUC** after fine-tuning, up from 82% accuracy when only the head is trained.

---

## 📌 Overview

Detecting accidents automatically from traffic camera feeds is a classic use case for lightweight, deployable computer vision models. This project fine-tunes **EfficientNetB0** (pretrained on ImageNet) on a CCTV accident-detection dataset, using:

- A small, dropout-regularized classification head on top of the frozen backbone
- On-the-fly data augmentation tailored to CCTV-style footage (flips, small rotations, zoom, translation, contrast jitter)
- Class weighting to correct for mild class imbalance
- A **two-stage training schedule**: train the head first, then unfreeze and fine-tune the top ~30% of the backbone at a low learning rate

The notebook covers the full pipeline: data loading → preprocessing → augmentation → model building → two-stage training → evaluation → single-image inference.

## 🗂️ Dataset

[Accident Detection from CCTV Footage](https://www.kaggle.com/datasets/ckay16/accident-detection-from-cctv-footage) (Kaggle, via `kagglehub`), split into `train` / `val` / `test`, each containing two classes:

| Split | Accident | Non Accident | Total |
|-------|---------:|--------------:|------:|
| Train | 369 | 422 | 791 |
| Validation | 46 | 52 | 98 |
| Test | 47 | 53 | 100 |

Images are resized to **224×224** and loaded via OpenCV (BGR → RGB conversion). The class split is only mildly imbalanced, but class weights are computed anyway since they cost nothing.

## 🏗️ Model Architecture

```
Input (224, 224, 3)
      │
Data Augmentation (train-time only)
      │
EfficientNetB0 (ImageNet weights, include_top=False)
      │
GlobalAveragePooling2D
      │
Dropout(0.35)
      │
Dense(128, activation="relu")
      │
Dropout(0.25)
      │
Dense(2, activation="softmax")
```

**Augmentation layers** (applied only during training, as part of the model graph):
- `RandomFlip("horizontal")`
- `RandomRotation(0.1)`
- `RandomZoom(0.1)`
- `RandomTranslation(0.05, 0.05)`
- `RandomContrast(0.1)`

## 🎯 Training Strategy — Two Stages

### Stage 1 — Frozen backbone
The EfficientNetB0 backbone is frozen; only the new head is trained.

| Setting | Value |
|---|---|
| Optimizer | Adam, `lr=3e-4` |
| Loss | Sparse categorical crossentropy |
| Epochs (max) | 25 |
| Callbacks | `EarlyStopping` (patience 5, from epoch 3), `ReduceLROnPlateau` (factor 0.5, patience 2) |

### Stage 2 — Fine-tuning
The backbone is unfrozen, but only the **top 30% of layers** are made trainable (BatchNorm layers stay frozen throughout to preserve pretrained statistics) — 57 of 238 EfficientNetB0 layers become trainable.

| Setting | Value |
|---|---|
| Optimizer | Adam, `lr=1e-5` |
| Loss | Sparse categorical crossentropy |
| Epochs (max) | 20 |
| Callbacks | `EarlyStopping` (patience 6), `ReduceLROnPlateau` (factor 0.5, patience 3), `ModelCheckpoint` (best `val_accuracy`) |

Both stages use the same `class_weight` and `batch_size=16`.

## 📊 Results

| Metric | Stage 1 (head only) | Stage 2 (fine-tuned) |
|---|---:|---:|
| Test accuracy | 0.82 | **0.92** |
| Test macro F1 | 0.82 | **0.92** |
| Test ROC-AUC | — | **0.955** |

Fine-tuning the top of the backbone gives a large jump in both accuracy and F1 over training the head alone. Full classification reports, the accuracy/loss curves (with a marker at the fine-tuning transition), the ROC curve, and a confusion matrix are generated in the notebook.

## 🔍 Inference

The notebook includes a ready-to-use `predict_image(filepath)` helper that loads an image, applies the same EfficientNet preprocessing, and returns the predicted class with a confidence score:

```python
predict_image("path/to/frame.jpg")
# Prediction: Accident
# Confidence: 91.4%
```

## 🛠️ Requirements

```
tensorflow>=2.x
opencv-python
numpy
pandas
scikit-learn
matplotlib
seaborn
kagglehub
```

## ▶️ Running the Notebook

1. Install the dependencies above (a GPU runtime, e.g. Google Colab or Kaggle, is strongly recommended).
2. Run the notebook top to bottom — the dataset is fetched automatically via `kagglehub.dataset_download("ckay16/accident-detection-from-cctv-footage")`.
3. Stage 1 trains the head; Stage 2 fine-tunes the backbone and saves the best model to `best_crash_detection_efficientnetb0.keras`.
4. Use the evaluation cells to reproduce the classification reports, ROC curve, and confusion matrix, or the `predict_image` cell to run inference on a single frame.

## 📁 Project Structure

```
.
├── efficientnet_headRegularization_headsize.ipynb   # Full pipeline: data → training → evaluation → inference
└── best_crash_detection_efficientnetb0.keras         # Saved best checkpoint (generated after Stage 2)
```

## 🔮 Possible Extensions

- Sweep head size / dropout rate systematically to quantify their effect on generalization (this notebook is one point in that sweep)
- Try other EfficientNet variants (B1–B7) or alternative backbones (ResNet, ConvNeXt)
- Extend from single-frame classification to short video-clip classification (temporal context)
- Export to TensorFlow Lite / ONNX for edge deployment on real CCTV hardware

## 📚 References

- [TensorFlow: Transfer Learning and Fine-Tuning](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [TensorFlow Guide: Transfer Learning & Fine-Tuning](https://www.tensorflow.org/guide/keras/transfer_learning)
- [Keras Applications Overview](https://keras.io/api/applications/)
- [EfficientNetB0 API Reference (Keras)](https://keras.io/api/applications/efficientnet/efficientnet_models/#efficientnetb0-function)
- [EfficientNet preprocess_input](https://www.tensorflow.org/api_docs/python/tf/keras/applications/efficientnet/preprocess_input)
- [Keras Preprocessing Layers — Image Augmentation](https://keras.io/api/layers/preprocessing_layers/image_augmentation/)
- [EarlyStopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping) · [ReduceLROnPlateau](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ReduceLROnPlateau) · [ModelCheckpoint](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint)

## 📄 License

Add a license of your choice (e.g. MIT) here before publishing.
