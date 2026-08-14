from __future__ import annotations
from pathlib import Path

import torch
from PIL import Image

from torch.utils.data import DataLoader, Dataset

from helpers.augmentations import augment_sample
from helpers.preprocess import load_yolo_boxes, preprocess, remap_bbox

class FireDetectionDataset(Dataset):
    def __init__(self, data_root, split, image_size = 640, augment=False, use_clahe=False):
        self.image_dir = Path(data_root) / split / "images"
        self.label_dir = Path(data_root) / split / "labels"

        self.image_path = sorted(self.image_dir.glob("*.jpg"))

        self.image_size = image_size
        self.augment = augment
        self.use_clahe = use_clahe
        if not self.image_path:
            raise FileNotFoundError(f"Veri bulunamadı: {self.image_dir}")

    def __len__(self):
            return len(self.image_path)

    def __getitem__(self, index):
            image_path = self.image_path[index]
            label_path = self.label_dir / f"{image_path.stem}.txt"

            image = Image.open(image_path).convert("RGB")

            boxes = load_yolo_boxes(label_path) if label_path.exists() else []

            if self.augment:
                image, boxes = augment_sample(image, boxes, use_clahe = self.use_clahe)

            width, height = image.size
            tensor, scale, pad_x, pad_y = preprocess(image, self.image_size)

            remapped = [
                [float(c), *remap_bbox(xc, yc, w, h, width, height,
                                    scale, pad_x, pad_y, self.image_size)]
                for c, xc, yc, w, h in boxes
            ]
            remapped = [b for b in remapped if b[3] > 0 and b[4] > 0]

            targets = (torch.tensor(remapped, dtype=torch.float32)
                    if remapped else torch.zeros((0, 5), dtype=torch.float32))
            meta = {"image_id": image_path.stem, "image_path": str(image_path),
                    "orig_width": width, "orig_height": height,
                    "scale": float(scale), "pad_x": int(pad_x), "pad_y": int(pad_y)}
            return torch.from_numpy(tensor), targets, meta

def build_dataloader(config, split):
    is_train = split == "train"

    dataset = FireDetectionDataset(
          config.data_root,split, config.image_size,
          augment= is_train and config.augmentation_enabled,
          use_clahe = config.use_clahe,)
    return DataLoader(
        dataset, batch_size=config.batch_size, shuffle=is_train,
        num_workers=config.num_workers, collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(), drop_last=is_train,
        persistent_workers=config.num_workers > 0,)