from pathlib import Path
import numpy as np
from PIL import Image
IMG_SIZE = 640

def resize_image(img: Image.Image, target_size: int = IMG_SIZE):
    # resize image while maintaining aspect ratio. Afterwards pad to square.
    orig_width, orig_height = img.size
    scale = min(target_size / orig_width, target_size / orig_height)
    new_width, new_height = round(orig_width * scale), round(orig_height * scale)

    resized = img.resize((new_width, new_height), Image.BICUBIC) #BICUBIC is a good choice for resizing images as we learned in class.

    canvas = Image.new("RGB", (target_size, target_size), (114, 114, 114)) # Standard in ultralytics documents
    pad_x = (target_size - new_width) // 2
    pad_y = (target_size - new_height) // 2
    canvas.paste(resized, (pad_x, pad_y))

    return canvas, scale, pad_x, pad_y


def remap_bbox(xc, yc, w, h, orig_width, orig_height, scale, pad_x, pad_y, target_size=IMG_SIZE):
    abs_xc, abs_yc = xc * orig_width, yc * orig_height
    abs_w, abs_h = w * orig_width, h * orig_height

    new_xc = abs_xc * scale + pad_x
    new_yc = abs_yc * scale + pad_y
    new_w = abs_w * scale
    new_h = abs_h * scale

    new_xc /= target_size
    new_yc /= target_size
    new_w /= target_size
    new_h /= target_size

    return new_xc, new_yc, new_w, new_h

def normalize_image(img: Image.Image):
    """Convert a PIL image to a float tensor in [0, 1] with shape (C, H, W)."""
    arr = np.asarray(img).astype(np.float32) / 255.0  # H, W, C
    tensor = np.transpose(arr, (2, 0, 1))             # C, H, W
    return tensor

def preprocess(img: Image.Image, target_size: int = IMG_SIZE):
    canvas, scale, pad_x, pad_y = resize_image(img, target_size)
    tensor = normalize_image(canvas)
    return tensor, scale, pad_x, pad_y


def load_yolo_boxes(label_path: Path): # Came from notebook 04
    text = label_path.read_text().strip()
    boxes = []
    for line in text.splitlines():
        cls_id, xc, yc, w, h = map(float, line.split())
        boxes.append((int(cls_id), xc, yc, w, h))
    return boxes