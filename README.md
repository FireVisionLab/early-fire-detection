# Early Fire & Smoke Detection

Object detection system for spotting small flames and early-stage smoke in indoor security camera footage. The goal is early detection, before a fire has a chance to spread, using two independently implemented and compared approaches.

## Approaches

Both models share the same dataset, preprocessing pipeline (640x640 letterbox), and evaluation module, so the comparison is apples-to-apples:

1. **EYT-Net** — a single-stage detector built from scratch in PyTorch: custom CNN backbone, two-scale detection head (P4/P5), IoU-based k-means anchors, CIoU box loss + BCE objectness/classification loss, and a full training loop with logging, checkpointing, and early stopping.
2. **Faster R-CNN** — a two-stage detector using a `torchvision` ResNet-50-FPN-v2 backbone pretrained on COCO, fine-tuned for the `fire` / `smoke` classes with a staged unfreezing schedule and a short hyperparameter search (learning rate, backbone layers, LR schedule).

## Results (test set, 1,300 images)

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F2 | Speed (RTX 4060) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Faster R-CNN** | **0.931** | **0.589** | **0.877** | **0.918** | **0.909** | ~95.7 ms (10.4 FPS) |
| **EYT-Net** | 0.365 | 0.170 | 0.413 | 0.453 | 0.444 | **~30.6 ms (32.6 FPS)** |

The pretrained two-stage model is clearly more accurate, especially on the harder `smoke` class. The from-scratch model is roughly 3x faster and much smaller (~7.2M vs ~43M parameters), trading accuracy for speed and independence from pretrained weights.

## Dataset

[Home Fire Dataset](https://github.com/PengBo0/Home-fire-dataset) — 6,500 images, 2 classes (`fire`, `smoke`), split 3,900 / 1,300 / 1,300 (train/val/test). Downloaded automatically via `kagglehub` in `notebooks/01_setup.ipynb`.

## Setup

Managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync
.venv\Scripts\activate
```

For GPU support, match your CUDA version in `pyproject.toml` (`[[tool.uv.index]]` URL, e.g. `cu128`/`cu130`), then re-run `uv lock && uv sync`. Verify with:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Project structure

```
data/          datasets (train/val/test images + YOLO-format labels)
eytnet/        EYT-Net model, dataset, loss, postprocessing, metrics
helpers/       shared preprocessing, augmentation, viz, training utils
notebooks/     data analysis, training, tuning, evaluation, demo
models/        checkpoints and run configs
```

Notebooks run roughly in order: data setup/EDA → preprocessing & augmentation → anchor analysis → Faster R-CNN fine-tuning & tuning → EYT-Net training → comparison, error analysis, and demo.
