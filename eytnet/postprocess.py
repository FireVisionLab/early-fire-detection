from __future__ import annotations

import numpy as np
import torch

from helpers.preprocess import preprocess as letterbox

def decode_scale(raw, anchors_wh, stride):
    """(B,A,H,W,5+C) ham cikti -> (B,A,H,W,4) [xc,yc,w,h], 640'lik tuvalde piksel.

    x = (sigmoid(tx)*2 - 0.5 + hucre_x) * stride   -> ofset [-0.5, 1.5]
    w = (sigmoid(tw)*2)^2 * anchor_w               -> carpan [0, 4]
    """
    height, width = raw.shape[2], raw.shape[3]
    ys, xs = torch.meshgrid(torch.arange(height, device=raw.device, dtype=raw.dtype),
                            torch.arange(width, device=raw.device, dtype=raw.dtype),
                            indexing="ij")
    grid = torch.stack((xs, ys), dim=-1).view(1, 1, height, width, 2)
    anchors = anchors_wh.to(raw.device, raw.dtype).view(1, -1, 1, 1, 2)

    xy = (raw[..., 0:2].sigmoid() * 2.0 - 0.5 + grid) * stride
    wh = (raw[..., 2:4].sigmoid() * 2.0) ** 2 * anchors
    return torch.cat([xy, wh], dim=-1)

def xywh_to_xyxy(boxes):
    xc, yc, w, h = boxes.unbind(-1)
    return torch.stack([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], dim=-1)

def box_iou(a, b):
    """(N,4) x (M,4) xyxy -> (N,M) IoU matrisi."""
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)

    # Kesisim dikdortgeni: sag-alt kosenin min'i, sol-ust kosenin max'i
    top_left = torch.max(a[:, None, :2], b[None, :, :2])
    bottom_right = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (bottom_right - top_left).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter).clamp(min=1e-9)

def nms(detections, iou_threshold):
    """Greedy NMS. detections: (N,6) [x1,y1,x2,y2,score,cls]

    Sinif ayrimi icin kutulara sinif_id * 10000 ekliyoruz; farkli siniflarin
    kutulari uzayda hic kesismedigi icin tek gecis sinif bazli NMS'e esit.
    """
    if len(detections) == 0:
        return detections
    boxes = detections[:, :4] + detections[:, 5:6] * 10_000.0
    order = detections[:, 4].argsort(descending=True)
    keep = []

    while order.numel() > 0:
        best = order[0]
        keep.append(int(best))
        if order.numel() == 1:
            break
        ious = box_iou(boxes[best].unsqueeze(0), boxes[order[1:]])[0]
        order = order[1:][ious <= iou_threshold]   # cok ortusenleri at

    return detections[torch.tensor(keep, dtype=torch.long, device=detections.device)]

def anchors_to_tensors(config, device):
    return [torch.tensor(a, dtype=torch.float32, device=device) for a in config.anchors]

def postprocess(predictions, config, score_threshold=None, anchors_list=None):
    if score_threshold is None:
        score_threshold = config.validation_score_threshold

    if anchors_list is None:
        anchors_list = anchors_to_tensors(config, predictions[0].device)
    batch_size = predictions[0].shape[0]
    candidates = [[] for _ in range(predictions[0].shape[0])]

    for raw, anchors, stride in zip(predictions, anchors_list, config.strides):
        boxes = xywh_to_xyxy(decode_scale(raw, anchors, stride))
        boxes = boxes.clamp(0, config.image_size)
        scores = raw[..., 4].sigmoid().unsqueeze(-1) * raw[..., 5:].sigmoid()  # (B,A,H,W,C)

        for i in range(batch_size):
            mask = scores[i] > score_threshold          # (A,H,W,C)
            if not bool(mask.any()):
                continue
            a, gy, gx, cls = mask.nonzero(as_tuple=True)
            candidates[i].append(torch.cat([
                boxes[i][a, gy, gx],
                scores[i][mask].unsqueeze(1),
                cls.unsqueeze(1).to(boxes.dtype),
            ], dim=1))

    results = []
    for per_image in candidates:
        if not per_image:
            results.append(torch.zeros((0,6), device=predictions[0].device))
            continue
        detections = nms(torch.cat(per_image).float(), config.nms_iou_threshold)

        if len(detections) > config.max_detections:
            detections = detections[:config.max_detections]   # NMS zaten skora gore sirali
        results.append(detections)

    return results

def to_original(detections, meta):

    """ letterbox tuvalindeki tespitleri orijinal görüntü pikseline çevirir"""
    if len(detections) == 0:
        return detections
    d = detections.clone().float()
    d[:, [0, 2]] = ((d[:, [0, 2]] - meta["pad_x"]) / meta["scale"]).clamp(0, meta["orig_width"])
    d[:, [1, 3]] = ((d[:, [1, 3]] - meta["pad_y"]) / meta["scale"]).clamp(0, meta["orig_height"])   
    return d



@torch.no_grad()
def collect_predictions(model, loader, config, device, score_threshold=None):
    "modeli tüm split üzerinde kosturup tahminleri ortak formata çevirir"

    model.eval()
    anchors = anchors_to_tensors(config, device)
    floor = config.decode_score_floor if score_threshold is None else score_threshold
    predictions = {}

    for images, _targets, metas in loader:
        raw = model(images.to(device))
        detections_list = postprocess(raw, config, floor, anchors)
        for detections, meta in zip(detections_list, metas):
            detections = to_original(detections.cpu(), meta)
            predictions[meta["image_id"]] = (detections.numpy() if len(detections)
                                             else np.zeros((0, 6), dtype=np.float32))
            
    return predictions

@torch.no_grad()
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