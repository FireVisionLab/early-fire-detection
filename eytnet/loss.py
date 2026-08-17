from __future__ import annotations

import torch
import torch.nn as nn

from eytnet.postprocess import decode_scale, xywh_to_xyxy, box_iou
def bbox_ciou(pred, target, eps=1e-9):
    pred_xyxy, target_xyxy = xywh_to_xyxy(pred), xywh_to_xyxy(target)

    # normal iou
    top_left = torch.max(pred_xyxy[:, :2], target_xyxy[:, :2])
    bottom_right = torch.min(pred_xyxy[:, 2:], target_xyxy[:, 2:])
    wh = (bottom_right - top_left).clamp(min=0)
    inter = wh[:,0]*wh[:,1]
    union = pred[:,2]*pred[:,3] + target[:,2]*target[:,3] - inter
    iou = inter / union.clamp(min=eps)

    # merkez uzakligi
    center_dist = ((pred[:, 0] - target[:, 0]) ** 2 + (pred[:, 1] - target[:, 1]) ** 2)
    enclose_tl = torch.min(pred_xyxy[:, :2], target_xyxy[:, :2])
    enclose_br = torch.max(pred_xyxy[:, 2:], target_xyxy[:, 2:])
    enclose_wh = (enclose_br - enclose_tl).clamp(min=0)
    diagonal = enclose_wh[:, 0] ** 2 + enclose_wh[:, 1] ** 2

    atan_diff = (torch.atan(target[:,2] / target[:,3].clamp(min=eps))
                 - torch.atan(pred[:,2] / pred[:,3].clamp(min=eps)))
    v = (4 / torch.pi ** 2) * atan_diff **2

    with torch.no_grad():
        alpha = v / (1 - iou + v).clamp(min=eps)

    return iou - center_dist / diagonal.clamp(min=eps) - alpha * v

def build_targets(targets, config, device):
    scale_count = len(config.strides)
    assignments = [{"b": [], "a": [], "gy": [], "gx": [], "cls": [], "tbox": []}
                   for _ in range(scale_count)]

    if len(targets) == 0:
        return _stack(assignments, device)

    # tüm anchor'lari tek listeye diz

    flat_anchors, owner = [], []
    for scale_id, anchors in enumerate(config.anchors):
        for anchor_id, (w, h) in enumerate(anchors):
            flat_anchors.append([w, h])
            owner.append((scale_id, anchor_id))
    flat_anchors = torch.tensor(flat_anchors, dtype=torch.float32)

    targets= targets.cpu()
    boxes_px = targets[:, 2:6] * config.image_size

    # en/boy iou: tuluar ayni merkeze oturtulmus gibi düşünülür
    inter = (torch.min(boxes_px[:, None, 2], flat_anchors[None, :, 0])
             * torch.min(boxes_px[:, None, 3], flat_anchors[None, :, 1]))
    union = (boxes_px[:, 2] * boxes_px[:, 3])[:, None] + \
            (flat_anchors[:, 0] * flat_anchors[:, 1])[None, :] - inter
    best_anchor = (inter / union.clamp(min=1e-9)).argmax(dim=1)

    for row, anchor_index in zip(targets, best_anchor.tolist()):
        scale_id, anchor_id = owner[anchor_index]
        stride = config.strides[scale_id]
        grid = config.grid_sizes[scale_id]

        xc, yc = (
            float(row[2] * config.image_size),
            float(row[3] * config.image_size),
        )

        gx = min(int(xc / stride), grid - 1)
        gy = min(int(yc / stride), grid - 1)

        item = assignments[scale_id]
        item["b"].append(int(row[0]))
        item["a"].append(anchor_id)
        item["gy"].append(gy)
        item["gx"].append(gx)
        item["cls"].append(int(row[1]))
        item["tbox"].append([xc, yc, float(row[4]) *config.image_size, float(row[5]) * config.image_size])

    return _stack(assignments, device)

def _stack(assignments, device):
    """python listelerini tensore cevir"""

    out = []

    for item in assignments:
        out.append({
            "b": torch.tensor(item["b"], dtype=torch.long, device=device),
            "a": torch.tensor(item["a"], dtype=torch.long, device=device),
            "gy": torch.tensor(item["gy"], dtype=torch.long, device=device),
            "gx": torch.tensor(item["gx"], dtype=torch.long, device=device),
            "cls": torch.tensor(item["cls"], dtype=torch.long, device=device),
            "tbox": torch.tensor(item["tbox"], dtype=torch.float32, device=device).reshape(-1, 4),
        })

    return out


class EYTNetLoss(nn.Module):

    """total loss= lambda_box*CIoU + lambda_obj*BCE(objectness) + lambda_cls*BCE(cls)"""

    def __init__(self, config):
        super().__init__()
        self.config=config
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, predictions, targets):
        cfg = self.config
        device = predictions[0].device
        assignments = build_targets(targets, cfg, device)
        anchor_tensors = [torch.tensor(a, dtype=torch.float32, device=device)
                          for a in cfg.anchors]

        box_loss = torch.zeros(1, device=device)
        obj_loss = torch.zeros(1, device=device)
        cls_loss = torch.zeros(1, device=device)
        positive_count = 0

        for raw, assign, anchors, stride in zip(predictions, assignments, 
                                                anchor_tensors, cfg.strides):
            decoded = decode_scale(raw, anchors, stride)

            obj_target = torch.zeros_like(raw[..., 4])
            obj_weight = torch.full_like(raw[..., 4], cfg.negative_objectness_weight)

            b, a, gy, gx = assign["b"], assign["a"], assign["gy"], assign["gx"]

            if len(b):
                positive_count += len(b)

                # kutu kaybi (CIoU)

                ciou = bbox_ciou(decoded[b, a, gy, gx], assign["tbox"])
                box_loss = box_loss + (1.0 - ciou).mean()

                # objectness kaybi
                obj_target[b, a, gy, gx] = 1.0
                obj_weight[b, a, gy, gx] = 1.0

                # Sinif kaybi 
                class_logits = raw[b, a, gy, gx, 5:]
                one_hot = torch.zeros_like(class_logits)
                one_hot[torch.arange(len(b), device=device), assign["cls"]] = 1.0
                cls_loss = cls_loss + self.bce(class_logits, one_hot).mean()                

            if len(targets):
                ignore = self._ignore_mask(decoded, targets, cfg, device)
                if len(b):
                    ignore[b, a, gy, gx] = False # pozitifler hep sayilir
                obj_weight[ignore] = 0.0

            obj_loss = obj_loss + (self.bce(raw[..., 4], obj_target)
                                   * obj_weight).sum() / obj_weight.sum().clamp(min=1)

        total = (cfg.lambda_box * box_loss
                 + cfg.lambda_objectness * obj_loss
                 + cfg.lambda_class * cls_loss)

        parts = {"box": float(box_loss.detach()), "obj": float(obj_loss.detach()),
                 "cls": float(cls_loss.detach()), "positives": positive_count}
        return total.squeeze(), parts

    @torch.no_grad()
    def _ignore_mask(self, decoded, targets, cfg, device):
        """GT ile IoU'su esiği aşan ama pozitif olmayan hücreleri işaretler"""

        batch_size = decoded.shape[0]
        mask = torch.zeros(decoded.shape[:4], dtype=torch.bool, device=device)
        decoded_xyxy = xywh_to_xyxy(decoded)

        for i in range(batch_size):
            rows = targets[targets[:, 0] == i]

            if len(rows) == 0:
                continue
            gt = xywh_to_xyxy(rows[:, 2:6].to(device) * cfg.image_size)
            best_iou = box_iou(decoded_xyxy[i].reshape(-1, 4), gt).max(dim=1).values
            mask[i] = (best_iou > cfg.ignore_iou_threshold).view(decoded.shape[1:4])
        return mask


  
            