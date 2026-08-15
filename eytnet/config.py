from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent 

class Config:

    def __init__(self, raw: dict, path: Path | None = None):
        self.raw, self.path = raw, path
        self.__dict__.update(raw)

        self.data_root = ROOT / raw["data_root"]
        self.run_dir = ROOT / raw["run_root"] / raw["experiment_name"]

        self.num_classes = len(self.class_names)
        self.grid_sizes = [self.image_size // s for s in self.strides]  
        self.anchors = self._load_anchors(ROOT / raw["anchors_path"])
        self.anchors_per_scale = len(self.anchors[0])

        if any(self.image_size % s for s in self.strides ):
            raise ValueError("image_size tüm stride'lara tam bölünmeli.")
        if not (self.data_root / "train" / "images").is_dir():
            raise FileNotFoundError(f"Veri kökü bulunamadı: {self.data_root}")

    @classmethod
    def load(cls, path) -> Config:
        path = Path(path)
        if not path.is_absolute():
            path = ROOT / path
        return cls(json.loads(path.read_text(encoding="utf-8")), path)
    
    def _load_anchors(self, anchors_path: Path) -> list[list[tuple[float,float]]]:
        "anchorlari alana göre sıralayıp scale'lara boler"

        data = json.loads(anchors_path.read_text(encoding="utf-8"))
        if data["img_size"] != self.image_size:
            raise ValueError(f"anchors.json {data['img_size']} px icin uretilmis, "
                             f"config image_size={self.image_size}.")

        boxes = sorted(((float(w), float(h)) for w, h in data["anchors_px"]), key=lambda wh: wh[0] * wh[1])
        n= len(boxes) // len(self.strides)

        return [boxes[i * n: (i + 1) * n] for i in range(len(self.strides))]


    def save(self) -> Path:
        "kullanılan configi runa kopyalar"

        self.run_dir.mkdir(parents=True, exist_ok=True)
        target = self.run_dir / "config.json"
        target.write_text(json.dumps(self.raw, indent=2), encoding="utf-8")
        return target

    def __repr__(self):
        return (f"Config({self.experiment_name}, {self.image_size}px, "
                f"grid={self.grid_sizes}, anchors={self.anchors})")


        