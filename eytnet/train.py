from __future__ import annotations

import math
import time

import torch

from helpers.utils import BestCheckpoint, append_metrics, load_checkpoint, set_seed
from eytnet.dataset import build_dataloader
from eytnet.loss import EYTNetLoss
from eytnet.metrics import evaluate_detections, load_ground_truth
from eytnet.model import build_model
from eytnet.postprocess import anchors_to_tensors, postprocess, to_original

def build_optimizer(model, config):
    """adam veya SGD, configten"""

    if config.optimizer.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                weight_decay=config.weight_decay)

    return torch.optim.SGD(model.parameters(), lr=config.learning_rate,
                           momentum=config.momentum, weight_decay=config.weight_decay,
                           nesterov=True)

def learning_rate_at(config, epoch, batch_index, batches_per_epoch):
    """warmup + cosine. adım basi cagrılır çünkü warmup epch icinde de artmali"""

    progress = epoch + batch_index / max(batches_per_epoch, 1)

    if progress < config.warmup_epochs:
        return config.learning_rate * (progress + 1e-8)/ config.warmup_epochs

    if config.lr_scheduler == "cosine":
        span = max(config.epochs - config.warmup_epochs, 1)
        ratio = (progress - config.warmup_epochs) / span
        cosine = 0.5 * (1 + math.cos(math.pi * min(ratio, 1.0)))
        final = config.learning_rate * config.final_lr_factor
        return final + (config.learning_rate - final) * cosine

    return config.learning_rate

def train_one_epoch(model, loader, loss_fn, optimizer, scaler, config, device, epoch, log_every=50):
    """tek epoch egitimi"""

    model.train()
    totals = {"loss" : 0.0, "box": 0.0, "obj": 0.0 ,"cls": 0.0}
    batches = len(loader)

    for i, (images, targets, _metas) in enumerate(loader):
        # ögrenme oranını adım basi güncelle

        lr = learning_rate_at(config, epoch, i, batches)
        for group in optimizer.param_groups:
            group["lr"] = lr

        images = images.to(device, non_blocking=True)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=config.use_amp):
            predictions = model(images)
            loss, parts = loss_fn(predictions, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)                       # clip'ten once olcegi geri al
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        totals["loss"] += float(loss.detach())
        for key in ("box", "obj", "cls"):
            totals[key] += parts[key]

        if log_every and (i + 1) % log_every == 0:
            print(f"  [{i + 1}/{batches}] loss={totals['loss'] / (i + 1):.4f} lr={lr:.5f}")

    return {k: v / max(batches, 1) for k, v in totals.items()}

@torch.no_grad()
def validate(model, loader, config, device, ground_truth):
    model.eval()
    anchors = anchors_to_tensors(config, device)
    predictions = {}

    for images, _targets, metas in loader:
        raw = model(images.to(device, non_blocking=True))
        for detections, meta in zip(postprocess(raw, config, config.decode_score_floor,
                                                anchors), metas):
            detections = to_original(detections.float().cpu(), meta)
            predictions[meta["image_id"]] = detections.numpy()

    return evaluate_detections(predictions, ground_truth, config.class_names,
                               score_threshold= config.validation_score_threshold,
                               map5095=False)

def train(config, device=None, resume=False):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config.seed)
    config.save()

    train_loader = build_dataloader(config, "train")
    val_loader = build_dataloader(config, "val")
    val_ground_truth = load_ground_truth(config.data_root, "val")

    model = build_model(config).to(device)
    loss_fn = EYTNetLoss(config)
    optimizer= build_optimizer(model, config)
    scaler = torch.amp.GradScaler(device.type, enabled = config.use_amp)
    checkpoint = BestCheckpoint(config.run_dir)

    start_epoch, best_map = 0, -1.0
    last_path = config.run_dir / "weights" / "last.pt"

    if resume and last_path.exists():
        state = load_checkpoint(last_path, model, optimizer)
        start_epoch = int(state["epoch"])

        best_path = config.run_dir / "weights" / "best.pt"

        if best_path.exists():
            best_state = torch.load(
                best_path,
                map_location="cpu",
                weights_only = False,
            )
            best_map = float(best_state.get("metric", -1.0)) 
        else:
            best_map = float(state.get("metric", -1.0))  # best.pt yoksa last.pt'den al

        checkpoint.best_metric = best_map
        
        print(
            f"Devam: epoch {start_epoch} sonrasindan, "
            f"en iyi mAP {best_map:.4f}"
        )

    print(f"{config.experiment_name} | {model.count_parameters():,} parametre | {device}")

    history, patience = [], 0
    for epoch in range(start_epoch, config.epochs):
        started = time.time()
        stats = train_one_epoch(model, train_loader, loss_fn, optimizer, scaler,
                                config, device, epoch)
        results = validate(model, val_loader, config, device, val_ground_truth)
        overall = results["overall"]

        row = {"epoch": epoch + 1, "train_loss": stats["loss"],
               "box": stats["box"], "obj": stats["obj"], "cls": stats["cls"],
               "map50": overall["map50"], "precision": overall["precision"],
               "recall": overall["recall"], "f1": overall["f1"], "f2": overall["f2"],
               "seconds": round(time.time() - started, 1)}

        history.append(row)
        append_metrics(config.run_dir / "metrics.csv", row)

        print(f"epoch {epoch + 1}/{config.epochs} | loss {stats['loss']:.4f} | "
              f"mAP@0.5 {overall['map50']:.4f} | F2 {overall['f2']:.4f} | {row['seconds']}s")

        # en iyi modeli mAP'e gore sakla + early stopping

        checkpoint.update(model, optimizer, epoch+1 , overall["map50"])
        if overall["map50"] > best_map:
            best_map, patience = overall["map50"], 0
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                print(f"Erken durdurma: {patience} epoch iyilesme yok.")
                break

    print(f"En iyi validation mAP@0.5: {best_map:.4f}")
    print(f"Agirliklar: {config.run_dir / 'weights' / 'best.pt'}")
    return model, config.run_dir, history

        