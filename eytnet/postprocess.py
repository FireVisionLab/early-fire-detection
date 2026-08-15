from __future__ import annotations
from curses import meta

import numpy as np
import torch

from helpers.preprocess import preprocess as letterbox

def decode_scale(raw: torch.Tensor, anchors_wh: torch.Tensor, stride: int) -> torch.Tensor:
    _, _, height, width, _ = raw.shape
    ys, xs = torch.meshgrid(torch.arange(height, device=raw.device, dtype=raw.dtype),
                            torch.arange(width, device=raw.device, dtype=raw.dtype),
                            indexing="ij")
    grid = torch.stack((xs, ys), dim=-1).view(1, 1, height, width, 2)
    anchors = anchors_wh.to(raw.device, raw.dtype).view(1, -1, 1, 1, 2)

    xy = (raw[..., 0:2].sigmoid() * 2.0 - 0.5 + grid) * stride
    wh = (raw[..., 2:4].sigmoid() * 2.0) ** 2 * anchors
    return torch.cat([xy, wh], dim=-1)

def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    xc, yc, w, h = boxes.unbind(-1)
    return torch.stack([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], dim=-1)

def box_iou(a: torch.Tensor, b:torch.Tensor) -> torch.Tensor:

    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)

    wh = (torch.min(a[:, None, 2:], b[:, 2:]) - torch.max(a[:, None, :2], b[None, :, :2])).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter).clamp(min=1e-9)

def nms(detections: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if len(detections) == 0:
        return detections
    boxes = detections[:, :4] + detections[:, 5:6] * 10_000.0
    scores = detections[:, 4]
    keep, order = [], scores.argsort(descending=True)

    while order.numel():
        keep.append(int(order[0]))
        if order.numel() == 1:
            break
        ious = box_iou(boxes[order[0]].unsqueeze(0), boxes[order[1:]]).squeeze(0)
        order = order[1:][ious <= iou_threshold]
    return detections[torch.tensor(keep, dtype= torch.long, device=detections.device)]

def anchors_to_tensors(config, device) -> list[torch.Tensor]:
    return [torch.tensor(a, dtype = torch.float32, devide = device) for a in config.anchors]

def postprocess(predictions, config, score_threshold=None, anchors_list=None):
    if score_threshold is None:
        score_threshold = config.validation_score_threshold

    if anchors_list is None:
        anchors_list = anchors_to_tensors(config, predictions[0].device)

    candidates = [[] for _ in range(predictions[0].shape[0])]

    for raw, anchors, stride in zip(predictions, anchors_list, config.strides):
        boxes = xywh_to_xyxy(decode_scale(raw, anchors, stride)).clamp(0, config.image_size)
        scores = raw[..., 4].sigmoid().unsqueeze(-1) * raw[..., 5:].sigmoid()  # (B,A,H,W,C)

        for b, score_map in enumerate(scores):
            mask = score_map > score_threshold
            if not bool(mask.any()):
                continue
            a, gy, gx, cls = mask.nonzero(as_tuple=True)
            candidates[b].append(torch.cat([boxes[b][a, gy, gx],
                                            score_map[mask].unsqueeze(1),
                                            cls.unsqueeze(1).to(boxes.dtype)], dim=1))

    results = []
    for per_image in candidates:
        if not per_image:
            results.append(torch.zeros((0,6), device=predictions[0].device))
            continue
        detections = nms(torch.cat(per_image).float(), config.nms_iou_threshold)

        if len(detections) > config.max_detections:
            detections = detections[detections[:4].argsort(descending=True)][:config.max_detections]
            results.append(detections)

    return results

def to_original(detections: torch.Tensor, meta: dict) -> torch.Tensor:

    """ letterbox tuvalindeki tespitleri orijinal görüntü pikseline çevirir"""
    if len(detections) == 0:
        return detections
    d = detections.clone().float()
    d[:, [0, 2]] = ((d[:, [0, 2]] - meta["pad_x"]) / meta["scale"]).clamp(0, meta["orig_width"])
    d[:, [1, 3]] = ((d[:, [1, 3]] - meta["pad_y"]) / meta["scale"]).clamp(0, meta["orig_height"])   
    return d

def predictions_to_dict(raw, config, metas, anchors, score_threshold):
    out = []

    for detections, meta in zip(postprocess(raw, config, score_threshold, anchors), metas):
        detections = to_original(detections.detach().cpu(), meta)
        out[meta["image_id"]] =(detections.numpy() if len(detections) else np.zeros((0,6), dtype=np.float32))

    return out

@torch.no_grad()
def collect_predictions(model, loader, config, device, score_threshold=None):
    "modeli tüm split üzerinde kosturup tahminleri ortak formata çevirir"

    model.eval()
    anchors = anchors_to_tensors(config, device)
    floor = config. decode_score_floor if score_threshold is None else score_threshold
    predictions = {}

    for images, _targets, matas in loader:
        raw = model(images.to(device, non_blocking=True))
        predictions.update(predictions_to_dict(raw,config, metas, anchors, floor))

    return predictions

def predict_image(model, image, config, device, score_threshold=None):
    "Tek bir PIL goruntusu icin uctan uca tahmin (demo icin)."
    model.eval()
    width, height = image.size
    tensor, scale, pad_x, pad_y = letterbox(image, config.image_size)
    raw = model(torch.from_numpy(tensor).unsqueeze(0).to(device))
    detections = postprocess(raw, config, score_threshold)[0]
    meta = {"orig_width": width, "orig_height": height,
            "scale": scale, "pad_x": pad_x, "pad_y": pad_y}
    return to_original(detections.cpu(), meta)