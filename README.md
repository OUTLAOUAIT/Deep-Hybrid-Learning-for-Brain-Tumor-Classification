# Brain Tumor Classification (PyTorch)

This repository provides a reproducible three-model classification project for the Figshare brain tumor dataset. The classes follow the notebook's implied mapping: `1 meningioma`, `2 glioma`, `3 pituitary`. The original notebook is preserved as `Hybrid_Deep_Learning_for_Brain_Tumor_classification.ipynb` for reference.

## What changed

The notebook augmented images before splitting, allowing an augmented image and its source to cross train/test boundaries. It also split without stratification or patient grouping, used a preprocessing path inconsistent with pretrained ImageNet weights, and evaluated only one architecture. The new pipeline splits source images first, performs augmentation only in the training dataset, makes validation/test transforms deterministic, uses validation macro-F1 for checkpoint selection, and evaluates test data through a separate command. The split utility groups declared subject IDs and exact duplicate files. Patient-level separation cannot be guaranteed when source MAT files do not provide a usable `PID`.

## Models

1. `efficientnet_b0`: ImageNet initialized EfficientNet-B0 with a replaceable classifier.
2. `efficientnet_vgg16_fusion`: EfficientNet-B0 and VGG16 feature maps are separately globally pooled and concatenated, followed by a compact trainable head. Both use the same ImageNet normalization.
3. `vit_b_16`: torchvision ViT-B/16 with an adapted classifier.

All models return logits for the shared cross-entropy training/evaluation pipeline. Freezing and pretrained weights are configurable. The fusion model is GPU-memory intensive; its example config freezes both backbones and uses batch size 8. ViT-B/16 is also substantial; lower batch size or backbone freezing is available in YAML. ImageNet weights download on first use when `pretrained: true`.

## Setup

Use Python 3.8–3.11 and a PyTorch/torchvision build appropriate for your platform and CUDA version. Install the pinned CPU-compatible package versions below, or replace the PyTorch wheels with the official CUDA-matched build for your machine.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The commands below assume the dependencies in `requirements.txt` are installed. Use a PyTorch/torchvision build compatible with your platform and CUDA version if you plan to train on a GPU.

## Data preparation

Download the [Figshare brain tumor dataset](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427) and place its extracted `*.mat` files under `data/raw/` (subdirectories are fine). It contains 3,064 T1-weighted contrast-enhanced images across meningioma, glioma, and pituitary classes. The preparation command extracts `cjdata.image`, maps the documented numeric labels, converts each image to per-image min-max grayscale PNG, and writes `data/processed/manifest.csv` plus train/val/test CSVs.

```bash
python -m scripts.prepare_data --config configs/base.yaml
```

The split is approximately 80/10/10 within classes at the independent group level. It uses `cjdata.PID` when present; otherwise exact file-byte duplicates are grouped. The script fails on conflicting labels within one group or too few independent groups per class. Inspect the resulting split counts and verify source-specific subject metadata before treating results as clinical evidence.

## Training

Run the same training engine with the requested model config:

```bash
python -m scripts.train --config configs/efficientnet_b0.yaml
python -m scripts.train --config configs/efficientnet_vgg16_fusion.yaml
python -m scripts.train --config configs/vit.yaml
```

Configurations define image size, batch size, epochs, learning rate, weight decay, seed, pretrained weights, freezing, dropout, AMP and checkpoint location. The trainer saves the best validation macro-F1 checkpoint and an epoch history JSON next to it. CUDA is selected when available; CPU is supported. Mixed precision is enabled only for CUDA.

## Evaluation and comparison

Evaluate each selected checkpoint exactly once on the test split. Reports include accuracy, macro/weighted F1, macro precision/recall, a confusion matrix, per-class report, and one-vs-rest macro ROC-AUC when defined.

```bash
python -m scripts.evaluate --config configs/efficientnet_b0.yaml --checkpoint outputs/checkpoints/efficientnet_b0.pt --output outputs/reports/efficientnet_b0.json
python -m scripts.evaluate --config configs/efficientnet_vgg16_fusion.yaml --checkpoint outputs/checkpoints/efficientnet_vgg16_fusion.pt --output outputs/reports/efficientnet_vgg16_fusion.json
python -m scripts.evaluate --config configs/vit.yaml --checkpoint outputs/checkpoints/vit_b_16.pt --output outputs/reports/vit_b_16.json
python -m scripts.compare_models --output-dir outputs/reports/
```

Comparison joins test reports to saved histories, ranks by the best validation macro-F1, and shows test metrics separately. It never creates missing results. Architecture-specific batch sizes and compute are not identical, so report training hardware, duration, and configuration alongside scores. Repeated seeds are not automated yet.

## Verification

Run `python -m unittest discover -s tests` after installing dependencies. The suite covers manifest loading and transforms, split reproducibility/duplicate separation, metrics, a tiny training/checkpoint smoke test, and all model output shapes. Real-dataset smoke tests remain to be performed because no MAT data or PyTorch runtime was available during this refactor. Full-scale training is intentionally not launched by the project commands.

## Limitations

- No dataset or verified patient metadata is committed. A patient-level split is possible only when `cjdata.PID` is present and correctly parsed.
- Split fallback detects byte-identical duplicates, not near-duplicates.
- The supplied notebook did not document an untouched subject-independent test cohort; this project establishes a new split, not an external validation.
- Transform choices and model hyperparameters are sensible starting points and require validation-only tuning.
- The pipeline currently reports training duration in the history file but does not benchmark inference latency or automatically generate plots.
- This is a research/engineering refactor and is not a medical diagnostic system.
