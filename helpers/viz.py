from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

CLASS_IDS = {
    0: {"name": "fire", "color": "red"},
    1: {"name": "smoke", "color": "blue"}
}


def draw_bbox(ax, label_path: Path, img_path: Path): 
    img_w, img_h = Image.open(img_path).size
    
    for line in label_path.read_text().strip().splitlines(): # THere can be multiple lines (multiple bounding boxes)
        if not line.strip():
            continue

        cls_id, xc, yc, w, h = map(float, line.split())
        x = (xc - w / 2) * img_w # Calculate top-left x coordinate of the bounding box
        y = (yc - h / 2) * img_h # Calculate top-left y coordinate of the bounding box

        rect = plt.Rectangle(
            (x, y), w * img_w, h * img_h,
            fill=False,
            edgecolor=CLASS_IDS[int(cls_id)]["color"],
            linewidth=2,
        )
        ax.add_patch(rect)

def draw_yolo_boxes(ax, boxes, img_w, img_h): 
    """Draw bounding boxxes based on img_w and img_h, which are the original image dimensions."""
    for box in boxes:
        cls_id, xc, yc, w, h = box
        x = (xc - w / 2) * img_w # Calculate top-left x coordinate of the bounding box
        y = (yc - h / 2) * img_h # Calculate top-left y coordinate of the bounding box

        rect = plt.Rectangle(
            (x, y), w * img_w, h * img_h,
            fill=False,
            edgecolor=CLASS_IDS[int(cls_id)]["color"],
            linewidth=2,
        )
        ax.add_patch(rect)