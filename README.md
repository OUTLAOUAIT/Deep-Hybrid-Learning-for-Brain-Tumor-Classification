# Brain Tumor Classification (PyTorch)

This repository provides a comparative brain tumor classification study using transfer learning on the Figshare dataset. It benchmarks VGG16, MobileNetV2, EfficientNet-B3, four Vision Transformer variants, and a ViT soft-voting ensemble. The original notebook is preserved as `Hybrid_Deep_Learning_for_Brain_Tumor_classification.ipynb` for reference.

## What changed

The notebook augmented images before splitting, allowing an augmented image and its source to cross train/test boundaries. It also split without patient grouping, used a preprocessing path inconsistent with pretrained ImageNet weights, and evaluated only one architecture. The refactored pipeline splits source images first, augments only training data, keeps validation/test transforms deterministic, uses validation accuracy and macro precision/recall for model selection, and evaluates the test split separately. Patient-level separation cannot be guaranteed when source MAT files do not provide a usable `PID`.

## Models

CNN comparison: `vgg16`, `mobilenet_v2`, and `efficientnet_b3`.

ViT comparison: `vit_b_16`, `vit_b_32`, `vit_l_16`, and `vit_l_32`, plus an equal-weight soft-voting ensemble of those four models.

Each individual model starts from ImageNet pretrained weights and fine-tunes the complete network (`pretrained: true`, `freeze_backbone: false`). The configs use inverse-frequency class-weighted cross-entropy. ViTs use the paper's `[-1, 1]` input scaling; CNNs use ImageNet normalization. Pretrained weights download on first use.

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

The class counts are 708 meningioma, 1,426 glioma, and 930 pituitary images. Training uses inverse-frequency class weights; validation and test reports include macro, weighted, and per-class precision/recall so overall accuracy cannot hide weak performance on a less common class.

```bash
python -m scripts.prepare_data --config configs/base.yaml
```

The split is approximately 80/10/10 within classes at the independent group level. It uses `cjdata.PID` when present; otherwise exact file-byte duplicates are grouped. The script fails on conflicting labels within one group or too few independent groups per class. Inspect the resulting split counts and verify source-specific subject metadata before treating results as clinical evidence.

## Training

Run each model on the same prepared train/validation/test split. In Colab, run these one at a time:

```bash
!python -m scripts.train --config configs/vgg16.yaml
!python -m scripts.train --config configs/mobilenet_v2.yaml
!python -m scripts.train --config configs/efficientnet_b3.yaml
!python -m scripts.train --config configs/vit_b_16.yaml
!python -m scripts.train --config configs/vit_b_32.yaml
!python -m scripts.train --config configs/vit_l_16.yaml --batch-size 2
!python -m scripts.train --config configs/vit_l_32.yaml --batch-size 2
```

Early stopping has a patience of six epochs. Checkpoints are selected using a validation score that equally averages accuracy, macro precision, and macro recall. Macro-averaging gives each tumor class equal weight despite the class imbalance. The test split is not used for checkpoint selection. Epoch history files record loss, accuracy, macro precision, macro recall, macro-F1, selection score, and training duration. Reduce batch size if Colab runs out of GPU memory.

## Evaluation and comparison

Evaluate each selected checkpoint once on the held-out test split. Reports include accuracy, balanced accuracy, macro and weighted precision/recall, macro and weighted F1, confusion matrix, and per-class precision/recall.

```bash
!python -m scripts.evaluate --config configs/vgg16.yaml --checkpoint outputs/checkpoints/vgg16.pt --output outputs/reports/vgg16.json
!python -m scripts.evaluate --config configs/mobilenet_v2.yaml --checkpoint outputs/checkpoints/mobilenet_v2.pt --output outputs/reports/mobilenet_v2.json
!python -m scripts.evaluate --config configs/efficientnet_b3.yaml --checkpoint outputs/checkpoints/efficientnet_b3.pt --output outputs/reports/efficientnet_b3.json
!python -m scripts.evaluate --config configs/vit_b_16.yaml --checkpoint outputs/checkpoints/vit_b_16.pt --output outputs/reports/vit_b_16.json
!python -m scripts.evaluate --config configs/vit_b_32.yaml --checkpoint outputs/checkpoints/vit_b_32.pt --output outputs/reports/vit_b_32.json
!python -m scripts.evaluate --config configs/vit_l_16.yaml --checkpoint outputs/checkpoints/vit_l_16.pt --output outputs/reports/vit_l_16.json
!python -m scripts.evaluate --config configs/vit_l_32.yaml --checkpoint outputs/checkpoints/vit_l_32.pt --output outputs/reports/vit_l_32.json
!python -m scripts.compare_models --output-dir outputs/reports/
```

The comparison CSV ranks individual models and the fixed ensemble using validation metrics only, then reports separate test results, including per-class precision and recall.

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
