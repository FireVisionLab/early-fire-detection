from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image

IOU_THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]   # 0.50 ... 0.95

def iou_matrix(a,b):
    """(N,4) x (M,4) xyxy --> (N,M) IoU matrisi."""

    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)

    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)

    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(bottom_right - top_left, 0, None)
    inter = wh[..., 0] * wh[..., 1]

    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)

def f_beta(precision, recall, beta):
    """Yangin tespitinde kacirilan alev, yanlis alarmdan daha maliyetli; bu yuzden
    F2'yi de raporluyoruz."""
    if precision + recall == 0:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * precision * recall / (b2 * precision + recall)

def average_precision(recall, precision):
    """PR eğrisinin altındaki alan
    precision sağdan sola monoton azalmayan hale getirilir, sonra recall'daki her artis o noktadaki precision ile çarpılıp toplanır"""

    if len(recall) == 0:
        return 0.0

    rec = np.concatenate(([0.0], recall, [1.0]))
    pre = np.concatenate(([1.0], precision, [0.0]))

    for i in range(len(pre) - 2, -1, -1):
        pre[i]= max(pre[i], pre[i + 1])

    steps = np.where(rec[1:] != rec[:-1])[0]

    return float(np.sum((rec[steps + 1] - rec[steps]) * pre[steps + 1]))

def prepare_class(predictions, ground_truths, class_id):
    """Bir sinif icin: skora gore sirali tahmin listesi + GT sayilari.

    Her tahmin icin o goruntudeki GT'lerle IoU satirini BIR KEZ hesaplayip
    saklıyoruz. mAP@0.5:0.95'te 10 esikte tekrar tekrar IoU hesaplamayalim diye. """
    entries = []
    gt_counts = {}

    for image_id, gt in ground_truths.items():
        gt = np.asarray(gt, np.float32).reshape(-1, 5)
        gt_boxes = gt[gt[:,4].astype(int) == class_id][:,:4]
        gt_counts[image_id] = len(gt_boxes)

        pred = predictions.get(image_id)

        if pred is None or len(pred) == 0:
            continue

        pred = np.asarray(pred, np.float32).reshape(-1, 6)
        pred = pred[pred[:,5].astype(int)==class_id]
        if len(pred) == 0:
            continue

        ious = iou_matrix(pred[:, :4], gt_boxes)
        for score, iou_row in zip(pred[:, 4], ious):
            entries.append((float(score), image_id, iou_row))

    entries.sort(key=lambda row: -row[0])          # yuksek guvenden dusuge
    return entries, gt_counts, sum(gt_counts.values())

def match(entries, gt_counts, iou_threshold, score_threshold=0.0):
    "her tahmin için TP mi?"

    used = {image_id: np.zeros(count, bool) for image_id, count in gt_counts.items()}
    is_tp = []

    for score, image_id, iou_row in entries:
        if score < score_threshold:
            break
        if len(iou_row) == 0:
            is_tp.append(False)
            continue
        row = np.where(used[image_id], -1.0, iou_row)
        best = int(np.argmax(row))

        if row[best] >= iou_threshold:
            used[image_id][best] = True
            is_tp.append(True)
        else:
            is_tp.append(False)

    return np.array(is_tp, bool)

def pr_curve(entries, gt_counts, total_gt, iou_threshold):
    """kümülatif TP/FT eğirisi"""
    if total_gt==0 or len(entries) == 0:
        return np.zeros(0), np.zeros(0)

    is_tp = match(entries, gt_counts, iou_threshold)
    tp = np.cumsum(is_tp)
    fp = np.cumsum(~is_tp)
    return tp/total_gt, tp / np.maximum(tp + fp, 1e-9)

def evaluate_detections(predictions, ground_truths, class_names,
                        score_threshold=0.25, iou_threshold=0.5, map5095=False):
    """tüm metrikleri hesaplar"""

    per_class , curves = {}, {}

    tp_all = fp_all =fn_all = 0

    for class_id, name in enumerate(class_names):
        entries, gt_counts, total_gt = prepare_class(predictions, ground_truths, class_id)

        recall, precision = pr_curve(entries, gt_counts, total_gt, iou_threshold)
        curves[name] = (recall, precision)
        ap50=average_precision(recall, precision)

        if map5095 and total_gt:
            ap5095 = float(np.mean([
                average_precision(*pr_curve(entries, gt_counts, total_gt, t))
                for t in IOU_THRESHOLDS
            ]))
        else:
            ap5095 = float("nan")

        # eşik uygulanmış sayimler (P / R / F1 / F2 için)
        is_tp = match(entries, gt_counts, iou_threshold, score_threshold)
        tp = int(is_tp.sum())
        fp = int(len(is_tp) - tp)
        fn = int(total_gt - tp)

        p = tp / max(tp + fp, 1e-9)
        r = tp / max(tp + fn, 1e-9)

        per_class[name] = {"ap50": ap50, "ap5095": ap5095,
                           "precision": p, "recall": r,
                           "f1": f_beta(p, r, 1), "f2": f_beta(p, r, 2),
                           "tp": tp, "fp": fp, "fn": fn, "support": total_gt}
        tp_all, fp_all, fn_all = tp_all + tp, fp_all + fp, fn_all + fn

    p = tp_all / max(tp_all + fp_all, 1e-9)
    r = tp_all / max(tp_all + fn_all, 1e-9)

    overall = {"map50": float(np.mean([m["ap50"] for m in per_class.values()])),
               "map5095": float(np.mean([m["ap5095"] for m in per_class.values()])),
               "precision": p, "recall": r,
               "f1": f_beta(p, r, 1), "f2": f_beta(p, r, 2),
               "tp": tp_all, "fp": fp_all, "fn": fn_all,
               "score_threshold": score_threshold}

    return {"per_class": per_class, "overall": overall, "curves": curves}


def confidence_sweep(predictions, ground_truths, class_names, thresholds=(0.05, 0.10, 0.15,0.20, 0.25, 0.30, 0.40, 0.50, 0.60)):
    """güven eşiği taraması. val üzerinden"""

    prepared = [prepare_class(predictions, ground_truths, c)
                for c in range(len(class_names))]

    rows = []
    for threshold in thresholds:
        tp = fp = fn = 0
        for entries, gt_counts, total_gt in prepared:
            is_tp = match(entries, gt_counts, 0.5, threshold)
            class_tp = int(is_tp.sum())

            tp+= class_tp
            fp+= int(len(is_tp) - class_tp)
            fn+= int(total_gt - class_tp)

        p = tp / max(tp + fp, 1e-9)
        r = tp / max(tp + fn, 1e-9)

        rows.append({"threshold": threshold, "precision": p, "recall": r,
                     "f1": f_beta(p, r, 1), "f2": f_beta(p, r, 2),
                     "tp": tp, "fp": fp, "fn": fn})
    return rows

def results_table(results):
    """Sonuc sozlugunu pandas tablosuna cevirir."""
    import pandas as pd

    rows = []
    for name, m in results["per_class"].items():
        rows.append({"sinif": name,
                     **{k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in m.items() if k != "support"}})
    o = results["overall"]
    rows.append({"sinif": "TOPLAM", "ap50": round(o["map50"], 4),
                 "ap5095": round(o["map5095"], 4),
                 "precision": round(o["precision"], 4), "recall": round(o["recall"], 4),
                 "f1": round(o["f1"], 4), "f2": round(o["f2"], 4),
                 "tp": o["tp"], "fp": o["fp"], "fn": o["fn"]})
    return pd.DataFrame(rows).set_index("sinif")


def load_ground_truth(data_root, split):
    """YOLO .txt etiketlerini --> {image_id: (M,5)} orijinal gorntü pikselinde"""

    image_dir = Path(data_root) / split / "images"
    label_dir = Path(data_root) / split / "labels"
    ground_truths = {}

    for image_path in sorted(image_dir.glob("*.jpg")):
        with Image.open(image_path) as img:
            width, height = img.size

        rows = []

        label_path = label_dir / f"{image_path.stem}.txt"

        if label_path.exists():
            for line in label_path.read_text().strip().splitlines():
                if not line.strip():
                    continue
                cls, xc, yc, w, h = map(float, line.split())
                xc, w = xc*width, w*width
                yc, h = yc*height, h*height

                rows.append([xc - w/2, yc - h / 2, xc + w/2, yc + h/2, cls])

        ground_truths[image_path.stem] = (np.array(rows, np.float32) if rows
                                          else np.zeros((0,5), np.float32))

    return ground_truths


