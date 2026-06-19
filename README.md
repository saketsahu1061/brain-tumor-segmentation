[README1.md](https://github.com/user-attachments/files/29135250/README1.md)
#  Brain Tumor Segmentation API

> A production-grade MLOps pipeline for automated brain tumor segmentation on grayscale MRI slices, powered by a ResNet34 U-Net architecture and served via a FastAPI inference gateway.

---

## Overview

This project delivers an end-to-end deep learning pipeline — from model training to REST API deployment — for segmenting brain tumors in clinical MRI scans. The model is built on the `segmentation-models-pytorch` library using a **ResNet34 encoder with a U-Net decoder**, trained on Kaggle's brain tumor segmentation dataset via Google Colab.

Key capabilities:

- Automated binary tumor mask prediction from grayscale MRI slices
- Incremental training with checkpoint-aware warm-start loading
- Interactive Swagger UI for real-time inference testing
- Dark-mode HTML home dashboard with hardware and model telemetry
- Containerized deployment via Docker

---

## Project Structure

```
brain-tumor-segmentation/
├── app/
│   ├── main.py              # FastAPI gateway with dark-mode HTML home UI
│   ├── predictor.py         # Inference wrapper (segmentation-models-pytorch)
│   └── preprocess.py        # Grayscale normalization & tensor pipeline
├── weights/
│   └── resnet34_brain_tumor.pth   # Trained model checkpoint
├── train/                   # Local training data directory
│   ├── _annotations.coco.json     # COCO-format polygon annotations
│   └── *.jpg                      # Raw clinical MRI scans
├── train.py                 # Incremental training engine
├── requirements.txt         # Pinned Python dependencies
├── Dockerfile               # Multi-stage container build
└── .dockerignore            # Build context exclusions
```

---

## Architecture

### Model

| Component | Detail |
|-----------|--------|
| Architecture | U-Net |
| Encoder | ResNet34 (ImageNet pretrained) |
| Input | Grayscale MRI slice — `[1, 640, 640]` |
| Output | Binary segmentation mask — `[1, 640, 640]` |
| Library | `segmentation-models-pytorch` |

### Loss Function

Training uses a combined objective that balances pixel-level accuracy with region overlap quality:

$$\mathcal{L} = \text{BCEWithLogitsLoss} + (1 - \text{Dice Coefficient})$$

The **Binary Cross-Entropy** term penalizes per-pixel misclassification, while the **Dice Loss** term maximizes volumetric overlap — critical for the class-imbalanced nature of tumor segmentation tasks.

---

## Training

### Data Format

The training engine expects data in **COCO JSON annotation format**. Set up your dataset directory as follows:

```
train/
├── _annotations.coco.json   # Polygon coordinate annotations (COCO format)
├── scan_001.jpg
├── scan_002.jpg
└── ...
```

The data loader parses polygon coordinate matrices on-the-fly and converts them into binary spatial masks of shape `[1, 640, 640]`.

### Incremental Training (Warm-Start)

The training engine is checkpoint-aware. When you run `python train.py`, it will:

1. **Scan** for an existing checkpoint at `weights/resnet34_brain_tumor.pth`
2. **Load** previously learned weights if found, preserving structural feature representations
3. **Fine-tune** using a conservative learning rate (`lr = 0.0001`) to prevent gradient instability
4. **Save** updated parameters back to disk after each training run

This design allows you to progressively improve the model across sessions without training from scratch.

```bash
python train.py
```

---

## Deployment

### Running the API Server

Start the Uvicorn ASGI gateway:

```bash
uvicorn app.main:app --reload
```

### Endpoints

| URL | Description |
|-----|-------------|
| `http://127.0.0.1:8000/` | Home dashboard — hardware status, model config, runtime telemetry |
| `http://127.0.0.1:8000/docs` | Swagger UI — upload MRI slices and inspect segmentation predictions |

### Docker

Build and run the containerized service:

```bash
docker build -t brain-tumor-seg .
docker run -p 8000:8000 brain-tumor-seg
```

---

## Inference Pipeline

The inference stack is cleanly modularized across three components:

- **`preprocess.py`** — Handles resizing, grayscale normalization, and tensor conversion (ensures values are in a valid positive range)
- **`predictor.py`** — Loads the checkpoint and wraps forward inference through the ResNet34 U-Net
- **`main.py`** — Exposes the prediction endpoint and renders the home dashboard UI

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies include:

- `fastapi` + `uvicorn` — API server
- `segmentation-models-pytorch` — U-Net + ResNet34 backbone
- `torch` + `torchvision` — Deep learning runtime
- `Pillow` + `numpy` — Image pre/post-processing
- `pycocotools` — COCO annotation parsing

---

## Dataset

The model weights were trained on the **Brain Tumor Segmentation** dataset available on [Kaggle](https://www.kaggle.com/), with training conducted on **Google Colab** using GPU acceleration. Annotations are stored in COCO polygon format and converted to binary masks during training.

---

## License

This project is released for research and educational use. Clinical deployment requires additional validation and regulatory compliance review.
