# 🫁 Chest X-Ray Pneumonia Classifier

A production-style deep learning pipeline for binary classification of chest X-rays (Normal vs. Pneumonia) using **PyTorch Lightning**, **transfer learning** on ResNet-18, and mixed-precision training.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Pipeline Design](#pipeline-design)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Inference](#inference)
- [Results](#results)
- [Project Structure](#project-structure)

---

## Overview

This project demonstrates an end-to-end medical image classification pipeline built with software engineering best practices. Rather than a simple notebook, the codebase is structured as a modular, reusable system that mirrors real-world ML deployment patterns:

- **Separation of concerns** — data, model, training, and inference are fully decoupled
- **Reproducibility** — seeded experiments and deterministic transforms
- **Efficiency** — mixed-precision (FP16) training, multi-worker DataLoaders with prefetching, and persistent workers
- **Robustness** — early stopping with configurable accuracy thresholds

---

## Architecture

```
Input (224×224 RGB X-ray)
        │
   ┌────▼────┐
   │ResNet-18│  ← Pre-trained backbone (all layers frozen)
   └────┬────┘
        │  512-dim feature vector
   ┌────▼────────────┐
   │ Linear(512 → 2) │  ← Fine-tuned classifier head
   └────┬────────────┘
        │
   Softmax → {NORMAL, PNEUMONIA}
```

**Transfer Learning Strategy:** Feature extraction — only the final fully connected layer (`model.fc`) has `requires_grad=True`. All backbone parameters are frozen. This is ideal for small medical datasets where catastrophic forgetting is a concern.

**Why ResNet-18?**

- Lightweight enough for fast iteration and deployment
- Deep residual connections prevent vanishing gradients
- Pre-trained ImageNet features generalise well to medical imaging textures

---

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — Kaggle

| Split | NORMAL | PNEUMONIA | Total |
| ----- | ------ | --------- | ----- |
| Train | 1,341  | 3,875     | 5,216 |
| Val   | 8      | 8         | 16    |
| Test  | 234    | 390       | 624   |

> **Note:** The validation set is intentionally small in the original dataset. Consider augmenting it by moving samples from the training set for more reliable evaluation.

---

## Pipeline Design

### Data Module (`ChestXRayDataModule`)

Inherits from `pl.LightningDataModule` to fully encapsulate the data lifecycle:

```python
dm = ChestXRayDataModule(data_dir="./chest_xray", batch_size=64)
dm.setup()
train_loader = dm.train_dataloader()
```

**Training augmentations applied:**

- Random horizontal flip
- Random rotation (±10°)
- Colour jitter (brightness + contrast ±0.2)
- Random affine: translation (±10%), scale (90–110%)
- Normalisation: mean `[0.482, 0.482, 0.482]`, std `[0.222, 0.222, 0.222]`

**DataLoader optimisations:**

- `num_workers=10` — parallel data loading
- `pin_memory=True` — faster CPU→GPU transfers
- `prefetch_factor=2` — prefetch next batch while GPU trains
- `persistent_workers=True` — avoid worker re-initialisation overhead

### Model (`ChestXRAYClassifier`)

Inherits from `pl.LightningModule`:

```python
model = ChestXRAYClassifier(
    model_weights_path="./resnet18_weights.pth",
    num_classes=2,
    learning_rate=1e-3,
    weight_decay=1e-2
)
```

**Optimizer:** AdamW with L2 regularisation  
**Scheduler:** `ReduceLROnPlateau` — reduces LR by 10× after 2 epochs of no improvement on `val_loss`

### Training (`run_training`)

```python
trainer, trained_model = run_training(
    model=model,
    data_module=dm,
    num_epochs=10,
    callback=early_stopping(10, target_accuracy=0.85)
)
```

**Trainer configuration:**

- `precision="16-mixed"` — AMP for ~2× speed on modern GPUs
- `accelerator="auto"` — automatically selects GPU/CPU
- `fast_dev_run=True` — dry-run mode for rapid iteration testing

---

## Installation

```bash
git clone https://github.com/yourusername/chest-xray-classifier.git
cd chest-xray-classifier

pip install torch torchvision lightning torchmetrics pillow matplotlib
```

**Requirements:**

- Python ≥ 3.9
- PyTorch ≥ 2.0
- PyTorch Lightning ≥ 2.0
- CUDA-compatible GPU (recommended)

---

## Usage

### Quick Start

```bash
# 1. Download dataset from Kaggle and extract to ./chest_xray/
# 2. Place your ResNet-18 weights at ./resnet18_chest_xray_classifier_weights.pth
# 3. Run training
python main.py
```

### Dry Run (smoke test — no GPU needed)

```python
from main import ChestXRayDataModule, ChestXRAYClassifier, early_stopping, run_training, setup_dummy_weights

weights = setup_dummy_weights("./dummy.pth", num_classes=2)
dm = ChestXRayDataModule("./chest_xray", batch_size=8)
model = ChestXRAYClassifier(weights, num_classes=2)
run_training(model, dm, num_epochs=1, callback=early_stopping(1, 0.99), dry_run=True)
```

---

## Training

Edit the `__main__` block in `main.py` to configure your run:

```python
pretrained_weights = "./resnet18_chest_xray_classifier_weights.pth"
training_epochs    = 10
target_accuracy    = 0.85

pl.seed_everything(15)  # Reproducibility

dm    = ChestXRayDataModule(data_dir, batch_size=64)
model = ChestXRAYClassifier(pretrained_weights, num_classes=2)

trainer, trained_model = run_training(
    model, dm, training_epochs,
    callback=early_stopping(training_epochs, target_accuracy)
)

torch.save(trained_model.model.state_dict(), "./trained_chest_xray_weights.pth")
```

---

## Inference

### Single Image

```python
from main import load_model, predict_image

model = load_model("./trained_chest_xray_weights.pth", num_classes=2)

result = predict_image(
    model,
    image_path="./test/NORMAL/IM-0003-0001.jpeg",
    class_names=["NORMAL", "PNEUMONIA"]
)

print(result["predicted_class"])   # → "NORMAL"
print(result["confidence"])        # → "97.43%"
print(result["all_scores"])        # → {"NORMAL": "97.43%", "PNEUMONIA": "2.57%"}
```

### Batch Folder

```python
from main import load_model, predict_folder

model = load_model("./trained_chest_xray_weights.pth", num_classes=2)
predict_folder(model, folder_path="./test/PNEUMONIA/", class_name=["NORMAL", "PNEUMONIA"])
```

---

## Results

| Metric               | Value                             |
| -------------------- | --------------------------------- |
| Final Val Accuracy   | ~0.875                            |
| Training Precision   | FP16 mixed                        |
| Early Stopping       | ✅ (patience = `num_epochs // 2`) |
| Backbone             | ResNet-18 (frozen)                |
| Trainable Parameters | ~1,026 (fc layer only)            |

---

## Project Structure

```
chest-xray-classifier/
│
├── main.py                              # Full pipeline
│   ├── display_dataset_count()          # Dataset statistics
│   ├── display_random_images()          # Visual EDA
│   ├── ChestXRayDataModule              # LightningDataModule
│   ├── ChestXRAYClassifier              # LightningModule
│   ├── run_training()                   # Trainer setup & fit
│   ├── load_model() / predict_image()   # Inference utilities
│   └── setup_dummy_weights()            # Testing helpers
│
├── chest_xray/
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── val/
│   └── test/
│
├── resnet18_chest_xray_classifier_weights.pth   # Pre-trained backbone
└── trained_chest_xray_weights.pth               # Your fine-tuned weights
```

---

## Key Design Decisions

**Why freeze the backbone?** With only ~5,200 training images, fine-tuning all 11M ResNet-18 parameters risks overfitting. Freezing the backbone and training only the 1,026-parameter head acts as strong implicit regularisation.

**Why AdamW over Adam?** AdamW decouples weight decay from the gradient update, leading to better generalisation — particularly important in small-dataset regimes.

**Why `ReduceLROnPlateau`?** Medical imaging models often plateau early. Automatically reducing the LR when validation loss stagnates squeezes out additional accuracy without manual intervention.

---

## Acknowledgements

- Dataset: [Paul Mooney — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Original paper: _Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning_ (Kermany et al., Cell 2018)
