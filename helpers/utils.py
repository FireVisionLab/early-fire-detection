import random
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import csv


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