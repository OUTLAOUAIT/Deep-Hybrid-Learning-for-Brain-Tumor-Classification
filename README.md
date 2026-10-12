# Brain Tumor Classification (PyTorch)

This repository provides a reproducible brain tumor classification project for the Figshare brain tumor dataset. The classes follow the notebook's implied mapping: `1 meningioma`, `2 glioma`, `3 pituitary`. The original notebook is preserved as `Hybrid_Deep_Learning_for_Brain_Tumor_classification.ipynb` for reference.

## What changed

The notebook augmented images before splitting, allowing an augmented image and its source to cross train/test boundaries. It also split without stratification or patient grouping, used a preprocessing path inconsistent with pretrained ImageNet weights, and evaluated only one architecture. The new pipeline splits source images first, performs augmentation only in the training dataset, makes validation/test transforms deterministic, uses validation macro-F1 for checkpoint selection, and evaluates test data through a separate command. The split utility groups declared subject IDs and exact duplicate files. Patient-level separation cannot be guaranteed when source MAT files do not provide a usable `PID`.

## Models

1. `efficientnet_b0`: ImageNet initialized EfficientNet-B0 with a replaceable classifier.
2. `efficientnet_vgg16_fusion`: EfficientNet-B0 and VGG16 feature maps are separately globally pooled and concatenated, followed by a compact trainable head. Both use the same ImageNet normalization.
3. `vit_b_16`, `vit_b_32`, `vit_l_16`, and `vit_l_32`: the four ViT variants used in the paper, with pretrained backbones and task-specific classifiers.

All models return logits for the shared cross-entropy training/evaluation pipeline. Freezing and pretrained weights are configurable. The fusion model is GPU-memory intensive; its example config freezes both backbones and uses batch size 8. ViT-L variants need smaller Colab batches, especially at 384×384. Pretrained weights download on first use when `pretrained: true`.

## Setup

Use Python 3.9 or newer. Google Colab already provides a compatible PyTorch/torchvision pair for its runtime and CUDA version, so the requirements file leaves those packages unpinned and does not replace Colab's build. For a local environment, install PyTorch and torchvision together using the [official PyTorch install selector](https://pytorch.org/get-started/locally/), then install the remaining project dependencies:

```bash
# Google Colab:
!pip install -r requirements.txt

# Local environment:
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
# First install the matching torch/torchvision pair from pytorch.org.
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

## Soft-voting ensemble

The paper's ensemble consists of ViT-B/16, ViT-B/32, ViT-L/16, and ViT-L/32. It averages their softmax probabilities with equal weight, then predicts the class with the largest mean probability. This project provides these four model variants and uses the paper's `[-1, 1]` ViT input scaling. Train each checkpoint separately at the same resolution:

```bash
!python -m scripts.train --config configs/vit_b_16.yaml --image-size 224 --checkpoint outputs/checkpoints/vit_b_16_224.pt
!python -m scripts.train --config configs/vit_b_32.yaml --image-size 224 --checkpoint outputs/checkpoints/vit_b_32_224.pt
!python -m scripts.train --config configs/vit_l_16.yaml --image-size 224 --batch-size 2 --checkpoint outputs/checkpoints/vit_l_16_224.pt
!python -m scripts.train --config configs/vit_l_32.yaml --image-size 224 --batch-size 2 --checkpoint outputs/checkpoints/vit_l_32_224.pt
```

For 384×384, repeat with `--image-size 384` and `_384` checkpoint names; use `--batch-size 1` or `2` if Colab runs out of GPU memory. The paper used a TPU for high-resolution training. Evaluate the 224×224 checkpoints together:

```bash
!python -m scripts.ensemble --configs configs/vit_b_16.yaml configs/vit_b_32.yaml configs/vit_l_16.yaml configs/vit_l_32.yaml --checkpoints outputs/checkpoints/vit_b_16_224.pt outputs/checkpoints/vit_b_32_224.pt outputs/checkpoints/vit_l_16_224.pt outputs/checkpoints/vit_l_32_224.pt --image-size 224 --output outputs/reports/vit_ensemble_224.json
```

Use the `_384.pt` checkpoints, `--image-size 384`, and `vit_ensemble_384.json` for the high-resolution ensemble. The code uses PyTorch ports of the original ViT families, while the paper used `vit-keras`; its reported scores also used a different data split and training setup, so they should not be treated as expected results here. The script saves ensemble metrics and a confusion matrix.

## Verification

Run `python -m unittest discover -s tests` after installing dependencies. The suite covers manifest loading and transforms, split reproducibility/duplicate separation, metrics, a tiny training/checkpoint smoke test, and all model output shapes. Real-dataset smoke tests remain to be performed because no MAT data or PyTorch runtime was available during this refactor. Full-scale training is intentionally not launched by the project commands.

## Limitations

- No dataset or verified patient metadata is committed. A patient-level split is possible only when `cjdata.PID` is present and correctly parsed.
- Split fallback detects byte-identical duplicates, not near-duplicates.
- The supplied notebook did not document an untouched subject-independent test cohort; this project establishes a new split, not an external validation.
- Transform choices and model hyperparameters are sensible starting points and require validation-only tuning.
- The pipeline currently reports training duration in the history file but does not benchmark inference latency or automatically generate plots.
- This is a research/engineering refactor and is not a medical diagnostic system.
