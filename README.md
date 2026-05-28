<div align="center">
  <img src="figures/Logo_EAFIT.png" alt="Universidad EAFIT" width="180"/>
  <h1>deepfakes_hunter</h1>
  <p><strong>Face-swap deepfake detection for KYC pipelines.</strong></p>

  [![PyPI version](https://img.shields.io/pypi/v/deepfakes_hunter)](https://pypi.org/project/deepfakes_hunter/)
  [![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
</div>

---

## What is deepfakes_hunter?

`deepfakes_hunter` is a Python package that exposes a single, ready-to-use class — `DeepfakeDetector` — for classifying face images as **real** or **face-swapped (deepfake)**.

Under the hood it runs a **ResNet-50** backbone fine-tuned for binary deepfake detection, exported to ONNX for fast, framework-agnostic inference. The model was trained on a curated dataset of 67 528 synthetic face-swaps generated with InSwapper → CodeFormer → Real-ESRGAN, paired with real faces from four public collections (CelebA, Flickr-Faces-HQ, UTKFace, VGGFace2) processed through a biometric-quality ETL.

> Full methodology and training details → [MODEL_CARD.md](MODEL_CARD.md)

---

## Install

```bash
pip install deepfakes_hunter
```

The model weights are downloaded automatically on first use to `~/.deepfakes_hunter/deepfake_detector.onnx`. No manual download required.

---

## Quick start

```python
from PIL import Image
from deepfakes_hunter import DeepfakeDetector

detector = DeepfakeDetector()          # downloads model on first run

img = Image.open("selfie.jpg")
result = detector.predict(img)

print(result)
# {'score': 0.0021, 'pred': 0, 'label': 'real', 'threshold': 0.98}
```

### Predict on a real face vs a face-swap

```python
from PIL import Image
from deepfakes_hunter import DeepfakeDetector

detector = DeepfakeDetector(threshold=0.98)

real_face  = Image.open("real.jpg")
fake_face  = Image.open("faceswap.jpg")

for img, name in [(real_face, "real"), (fake_face, "faceswap")]:
    r = detector.predict(img)
    print(f"{name:>10}  score={r['score']:.4f}  label={r['label']}")

# output:
#       real  score=0.0031  label=real
#   faceswap  score=0.9987  label=deepfake
```

### Load your own model weights

```python
detector = DeepfakeDetector(onnx_path="path/to/your_model.onnx", threshold=0.98)
```

### GPU acceleration

```python
detector = DeepfakeDetector(use_gpu=True)   # requires onnxruntime-gpu
```

---

## API reference

### `DeepfakeDetector(onnx_path=None, threshold=0.98, use_gpu=False)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `onnx_path` | `str \| None` | `None` | Path to a local `.onnx` file. If `None`, the model is downloaded automatically. |
| `threshold` | `float` | `0.98` | Score ≥ threshold → predicted deepfake. |
| `use_gpu` | `bool` | `False` | Use `CUDAExecutionProvider` (requires `onnxruntime-gpu`). |

### `.predict(img_pillow, threshold=None) → dict`

| Key | Type | Description |
|-----|------|-------------|
| `score` | `float` | Sigmoid probability of being a deepfake (0 → real, 1 → deepfake). |
| `pred` | `int` | `1` if deepfake, `0` if real. |
| `label` | `str` | `"deepfake"` or `"real"`. |
| `threshold` | `float` | Threshold used for this prediction. |

---

## Pipeline overview

The model was trained through an end-to-end reproducible pipeline:

![End-to-end pipeline](figures/fig_pipeline_e2e.png)

| Stage | What happens |
|-------|-------------|
| **1 — Data** | Four public face datasets; RetinaFace detection + 6DRepNet pose filtering (ISO/IEC 19794-5) |
| **2 — Embeddings** | ArcFace 512-dim embeddings; P50 / P99 cosine-similarity pair selection |
| **3 — Face swaps** | 67 528 swaps with InSwapper → CodeFormer → Real-ESRGAN |
| **4 — Manual review** | FaceSwap Tagger web tool; 2 649 (8.27 %) artifacts discarded |
| **5 — Splits** | Grouped by `target_id`; 99:1 real:fake imbalance in val / test |
| **6 — Training** | ResNet-18 (baseline) and ResNet-50 (final) with identical fine-tuning policy |

---

## Performance (test set · 11 600 samples · 99:1 imbalance)

| Model | PR-AUC | F1 | Recall | Precision |
|-------|--------|-----|--------|-----------|
| **ResNet-50** (final) | **0.9977** | **0.983** | **0.991** | **0.975** |
| ResNet-18 (baseline) | 0.9978 | 0.970 | 0.983 | 0.957 |

Evaluated at threshold = 0.98 (optimal on validation).

---

## Dataset examples — real vs face-swap

<div align="center">

**Real faces (post-ETL)**

![Real face examples](figures/fig_dataset_examples_real.png)

**Face-swaps (InSwapper → CodeFormer → Real-ESRGAN)**

![Face-swap examples](figures/fig_dataset_examples_swap.png)

</div>

---

## Repository structure

The `src/` folder documents the full training pipeline as reproducible Jupyter notebooks:

```
src/
├── 01_data_preparation/
│   ├── 01_download_datasets.ipynb       ← download CelebA, Flickr, UTKFace, VGGFace2
│   ├── 02_etl_face_detection.ipynb      ← RetinaFace: detect, align, crop
│   ├── 03_etl_pose_filtering.ipynb      ← 6DRepNet: pose filter (ISO/IEC 19794-5)
│   └── 04_etl_demographics.ipynb        ← age & gender characterisation
│
├── 02_faceswap_generation/
│   ├── 01_arcface_embeddings.ipynb      ← extract ArcFace embeddings
│   ├── 02_pair_selection_p50_p99.ipynb  ← cosine-similarity P50/P99 pair selection
│   └── 03_inswapper_codeformer.ipynb   ← InSwapper → CodeFormer → Real-ESRGAN
│
├── 03_training/
│   ├── 01_dataset_assembly_splits.ipynb ← merge real+fake, build splits
│   ├── 02_train_resnet18.ipynb          ← ResNet-18 fine-tuning (baseline)
│   └── 03_train_resnet50.ipynb          ← ResNet-50 fine-tuning (final model)
│
├── 04_evaluation/
│   ├── 01_export_onnx.ipynb             ← export best checkpoint to ONNX
│   └── 02_test_inference.ipynb          ← threshold sweep, confusion matrix, PR curve
│
└── 05_interpretability/
    └── 01_gradcam_analysis.ipynb        ← Grad-CAM activation maps
```

---

## Authors

**Javier Javier Daza Olivella** · [jjdazao@eafit.edu.co](mailto:jjdazao@eafit.edu.co)

Advisor: **Pablo Andrés Saldarriaga** · [psaldar2@eafit.edu.co](mailto:psaldar2@eafit.edu.co)

Universidad EAFIT · Escuela de Ciencias Aplicadas e Ingeniería
Maestría en Ciencia de Datos y Analítica · Medellín, 2026

---

*MIT License*
