import random
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import csv
import json
import time
from eytnet.metrics_compat import evaluate_map


def set_seed(seed=42): # For reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_run_dir(base_dir, run_name):
    run_dir = Path(base_dir) / run_name
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    return run_dir

def save_checkpoint(path, model, optimizer, epoch, metric):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "metric": metric,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)

def load_checkpoint(path, model, optimizer = None):
    checkpoint = torch.load(Path(path))
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint

def append_metrics(csv_path, row):
    csv_path = Path(csv_path)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

class BestCheckpoint:
    # keeps last.pt and best.pt, similar to what YOLO does automatically
    def __init__(self, run_dir): 
        self.last_path = Path(run_dir) / "weights" / "last.pt"
        self.best_path = Path(run_dir) / "weights" / "best.pt"
        self.best_metric = float("-inf")

    def update(self, model, optimizer, epoch, metric):
        save_checkpoint(self.last_path, model, optimizer, epoch, metric)
        if metric > self.best_metric:
            self.best_metric = metric
            save_checkpoint(self.best_path, model, optimizer, epoch, metric)

def yolo_log_kwargs(project, name, save_period=5):
    # pass into model.train(...)
    return {
        "project": str(project),
        "name": name,
        "exist_ok": True,
        "save": True,
        "save_period": save_period,
    }

def read_yolo_metrics(run_dir):
    df = pd.read_csv(Path(run_dir) / "results.csv")
    df.columns = [c.strip() for c in df.columns]
    return df

def f2_score(p, r):
    if p + r == 0:
        return 0.0
    return (5 * p * r) / (4 * p + r)

def f1_score(p, r):
    if p + r == 0:
        return 0.0
    return (2 * p * r) / (p + r)

def save_run_config(run_dir, config):
    with (Path(run_dir) / "config.json").open("w") as f:
        json.dump(config, f, indent=2)

def train_detection_run(model, optimizer,train_loader, val_loader, device, run_dir, epochs, start_epoch=1, log_every = 200, scheduler=None, patience=None, min_epoch=1):
    ckpt = BestCheckpoint(run_dir)
    stale = 0

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        t0 = time.time()
        n = 0
        train_parts = {}
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for k, v in loss_dict.items():
                train_parts[k] = train_parts.get(k, 0.0) + v.item()
            train_parts["total"] = train_parts.get("total", 0.0) + loss.item()


            n += 1
            if n % log_every == 0:
                print(f"Epoch {epoch}, Step {n}, Loss: {train_parts['total'] / n}")

        train_parts = {f"train_{k}": v / n for k, v in train_parts.items()}

        val_parts = {}
        val_n = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                for k, v in loss_dict.items():
                    val_parts[k] = val_parts.get(k, 0.0) + v.item()
                val_parts["total"] = val_parts.get("total", 0.0) + sum(loss_dict.values()).item()
                val_n += 1
        val_parts = {f"val_{k}": v / val_n for k, v in val_parts.items()}

        map50, aps50 = evaluate_map(model, val_loader, device, iou_thresh=0.5)
        map75, aps75 = evaluate_map(model, val_loader, device, iou_thresh=0.75)

        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            **train_parts,
            **val_parts,
            "map50": map50,
            "map75": map75,
            "ap50_fire": aps50.get("fire"),
            "ap50_smoke": aps50.get("smoke"),
            "ap75_fire": aps75.get("fire"),
            "ap75_smoke": aps75.get("smoke"),
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_sec": time.time() - t0,
        }

        ckpt.update(model, optimizer, epoch, row["map50"])
        append_metrics(run_dir / "metrics.csv", row)
        print(
            f"Epoch {epoch}: val_total {row['val_total']}, "
            f"map50 {map50}, map75 {map75}, {row['epoch_sec']}s"
        )

        if patience is not None:
            if row["map50"] < ckpt.best_metric:
                stale += 1
            else:
                stale = 0
            if stale >= patience and epoch >= min_epoch:
                print(f"Early stopping at epoch {epoch}. Best map50: {ckpt.best_metric}")
                break

    return ckpt