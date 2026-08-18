from __future__ import annotations
import numpy as np
import torch

from PIL import Image

from eytnet.metrics import evaluate_detections

CLASS_NAMES = ["fire", "smoke"]
TV_TO_COMMON = {1: 0, 2: 1}          
COLLECT_FLOOR = 0.01                 # AP icin dusuk skorlu tahminler de lazim

def letterbox_meta(image_path, image_size=640):

    """dosyayi acmadan kutu çevirmek için gereken letterbox parameterleri"""

    with Image.open(image_path) as img:
        width, height = img.size

    scale = min(image_size / width, image_size / height)
    new_width, new_height = round(width * scale), round(height * scale)
    return {"orig_width": width, "orig_height": height, "scale": scale,
            "pad_x": (image_size - new_width) // 2,
            "pad_y": (image_size - new_height) // 2}

def to_original(boxes, meta):
    """letterbos tuvalindeki xyxy --> orijinal goruntu pikseli """

    if len(boxes) == 0:
        return boxes

    out = boxes.astype(np.float32).copy()
    out[:, [0,2]] = np.clip((out[:, [0,2]] , meta["pad_x"]) /meta["scale"],
                            0, meta["orig_width"])

    out[:, [1, 3]] = np.clip((out[:, [1, 3]] - meta["pad_y"]) / meta["scale"],
                             0, meta["orig_height"])
    return out

@torch.no_grad()
def collect_predictions(model, loader, device, score_threshold=COLLECT_FLOOR,
                        image_size=640):
    """faster R-CNN + (images,targets) lodar --> ortak format"""

    model.eval()
    paths = list(getattr(loader.dataset, "image_paths", []))
    predictions, ground_truths = {}, {}

    counter = 0

    for images, targets in loader:
        outputs = model([image.to(device) for image in images])

        for target, output in zip(targets, outputs):
            index = int(target["image_id"].item()) if "image_id" in target else counter
            counter += 1
            path = paths[index]
            image_id = path.stem
            meta = letterbox_meta(path, image_size)

            # tahminler 

            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            keep = [i for i in range(len(scores))
                    if scores[i] >= score_threshold and int(labels[i]) in TV_TO_COMMON]

            if keep:
                classes = np.array([[TV_TO_COMMON[int(labels[i])]] for i in keep], np.float32)
                predictions[image_id] = np.concatenate(
                    [to_original(boxes[keep], meta), scores[keep].reshape(-1, 1), classes],
                    axis=1).astype(np.float32)

            else:
                predictions[image_id] = np.zeros((0,6), np.float32)

            # Ground truth'lar

            gt_boxes = target["boxes"].cpu().numpy()
            gt_labels = target["labels"].cpu().numpy()
            keep = [i for i in range(len(gt_labels)) if int(gt_labels[i]) in TV_TO_COMMON]

            if keep:
                classes = np.array([[TV_TO_COMMON[int(gt_labels[i])]] for i in keep], np.float32)
                ground_truths[image_id] = np.concatenate(
                    [to_original(gt_boxes[keep], meta), classes], axis=1).astype(np.float32)

            else:
                ground_truths[image_id] = np.zeros((0,5), np.float32)

    return predictions, ground_truths


def evaluate_map(model, loader, device, iou_threshold = 0.5, score_thresh = 0.05):
    predictions, ground_truths = collect_predictions(model, loader, device)
    results = evaluate_detections(predictions, ground_truths, CLASS_NAMES,
                                  score_threshold=score_thresh, iou_threshold=iou_threshold, map5095=True)

    class_aps = {name: m["ap"] for name, m in results["per_class"].items()
                 if m["support"] > 0}
    for name, ap in class_aps.items():
        print(f"AP@{iou_threshold:.2f} {name}: {ap:.3f}")

    map_score = sum(class_aps.values()) / len(class_aps) if class_aps else 0.0
    print(f"mAP@{iou_threshold:.2f}: {map_score:.3f}")
    return map_score, class_aps

def evaluate_full(model, loader, device, score_thresh = 0.25):
    """Rapor sayilari icin: P/R/F1/F2 + mAP@0.5 + mAP@0.5:0.95 + PR egrileri."""
    predictions, ground_truths = collect_predictions(model, loader, device)
    return evaluate_detections(predictions, ground_truths, CLASS_NAMES,
                               score_threshold=score_thresh, map5095=True)