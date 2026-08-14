from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

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