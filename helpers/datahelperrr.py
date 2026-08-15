from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict
import numpy as np
from helpers.augmentations import augment_sample
from helpers.preprocess import (
    IMG_SIZE,
    load_yolo_boxes,
    preprocess,
    remap_bbox,
)

# torchvision treats 0 as background. Dataset files use 0=fire, 1=smoke.
YOLO_TO_TV = {0: 1, 1: 2}


def yolo_to_xyxy(boxes, orig_w, orig_h, scale, pad_x, pad_y, target_size=IMG_SIZE):
    """Normalize YOLO boxes -> xyxy pixels on the letterboxed canvas."""
    xyxy = []
    labels = []
    for cls_id, xc, yc, w, h in boxes:
        xc, yc, w, h = remap_bbox(
            xc, yc, w, h, orig_w, orig_h, scale, pad_x, pad_y, target_size
        )
        x1 = max(0.0, (xc - w / 2) * target_size)
        y1 = max(0.0, (yc - h / 2) * target_size)
        x2 = min(float(target_size), (xc + w / 2) * target_size)
        y2 = min(float(target_size), (yc + h / 2) * target_size)
        if x2 <= x1 or y2 <= y1:
            continue
        xyxy.append([x1, y1, x2, y2])
        labels.append(YOLO_TO_TV[int(cls_id)])
    return xyxy, labels


class FireDataset(Dataset):
    def __init__(self, split_dir: Path, augment: bool = False, img_size: int = IMG_SIZE):
        self.img_dir = Path(split_dir) / "images"
        self.label_dir = Path(split_dir) / "labels"
        self.augment = augment
        self.img_size = img_size
        self.image_paths = sorted(self.img_dir.glob("*.jpg"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_dir / f"{img_path.stem}.txt"

        img = Image.open(img_path).convert("RGB")
        boxes = load_yolo_boxes(label_path) if label_path.exists() else []

        if self.augment:
            img, boxes = augment_sample(img, boxes)

        orig_w, orig_h = img.size
        tensor, scale, pad_x, pad_y = preprocess(img, self.img_size)
        xyxy, labels = yolo_to_xyxy(
            boxes, orig_w, orig_h, scale, pad_x, pad_y, self.img_size
        )

        image = torch.from_numpy(tensor)
        target = {
            "boxes": torch.tensor(xyxy, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        return image, target


def collate_fn(batch):
    # torchvision detection models take a list of images, not a stacked batch
    return tuple(zip(*batch))


# ---------------------------------------------------------------------------
# Temporary detection metrics (numpy only, no torchmetrics/pycocotools).
# This is a placeholder for hyperparameter/architecture model selection.
# The shared P/R/F1/AP/mAP evaluation module is Ömer's part; delete this
# block once that module is ready and re-score the final checkpoint with it.
# ---------------------------------------------------------------------------

CLASS_NAMES = {1: "fire", 2: "smoke"}


def iou_xyxy(box, other):
    x1, y1 = max(box[0], other[0]), max(box[1], other[1])
    x2, y2 = min(box[2], other[2]), min(box[3], other[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_box = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_other = max(0.0, other[2] - other[0]) * max(0.0, other[3] - other[1])
    return inter / max(area_box + area_other - inter, 1e-9)


def average_precision(preds_by_image, gts_by_image, iou_thresh=0.5):
    n_gt = sum(len(v) for v in gts_by_image.values())
    if n_gt == 0:
        return None

    flat_preds = [(img_id, score, box) for img_id, items in preds_by_image.items() for score, box in items]
    flat_preds.sort(key=lambda row: -row[1])  # highest confidence first

    matched = {img_id: np.zeros(len(boxes), dtype=bool) for img_id, boxes in gts_by_image.items()}
    tp = np.zeros(len(flat_preds))
    fp = np.zeros(len(flat_preds))

    for i, (img_id, score, box) in enumerate(flat_preds):
        gt_boxes = gts_by_image.get(img_id, [])
        if not gt_boxes:
            fp[i] = 1
            continue
        ious = [iou_xyxy(box, gt) for gt in gt_boxes]
        best = int(np.argmax(ious))
        if ious[best] >= iou_thresh and not matched[img_id][best]:
            tp[i] = 1
            matched[img_id][best] = True  # each GT box can only be matched once
        else:
            fp[i] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / n_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

    # standard AP: precision envelope must be non-increasing as recall grows
    envelope = precision.copy()
    for i in range(len(envelope) - 2, -1, -1):
        envelope[i] = max(envelope[i], envelope[i + 1])

    recall_pad = np.concatenate(([0.0], recall))
    precision_pad = np.concatenate(([envelope[0] if len(envelope) else 0.0], envelope))
    ap = float(np.sum((recall_pad[1:] - recall_pad[:-1]) * precision_pad[1:]))
    return ap, precision, recall


@torch.no_grad()
def collect_val_predictions(model, loader, device, score_thresh=0.05):
    model.eval()
    preds_by_class = defaultdict(lambda: defaultdict(list))
    gts_by_class = defaultdict(lambda: defaultdict(list))

    img_id = 0
    for images, targets in loader:
        images_dev = [img.to(device) for img in images]
        outputs = model(images_dev)

        for target, output in zip(targets, outputs):
            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            for box, score, label in zip(boxes, scores, labels):
                if score < score_thresh:
                    continue
                preds_by_class[int(label)][img_id].append((float(score), box))

            for box, label in zip(target["boxes"].numpy(), target["labels"].numpy()):
                gts_by_class[int(label)][img_id].append(box)

            img_id += 1

    return preds_by_class, gts_by_class


def evaluate_map(model, loader, device, iou_thresh=0.5, score_thresh=0.05):
    preds_by_class, gts_by_class = collect_val_predictions(model, loader, device, score_thresh)

    class_aps = {}
    for cls_id, name in CLASS_NAMES.items():
        result = average_precision(preds_by_class[cls_id], gts_by_class[cls_id], iou_thresh)
        if result is None:
            continue
        ap, _, _ = result
        class_aps[name] = ap
        print(f"AP@{iou_thresh:.2f}  {name}: {ap:.3f}")

    map_score = sum(class_aps.values()) / len(class_aps)
    print(f"mAP@{iou_thresh:.2f}: {map_score:.3f}")
    return map_score, class_aps