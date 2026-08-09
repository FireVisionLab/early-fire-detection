import random
import cv2
import numpy as np
from PIL import Image, ImageEnhance


def horizontal_flip(img: Image.Image, boxes, p=0.5):
    if random.random() >= p:
        return img, boxes
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    boxes = [(cls_id, 1.0 - xc, yc, w, h) for cls_id, xc, yc, w, h in boxes]
    return img, boxes


def limited_color_jitter(img: Image.Image, brightness=0.15, contrast=0.15, saturation=0.15):
    img = ImageEnhance.Brightness(img).enhance(random.uniform(1 - brightness, 1 + brightness))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(1 - contrast, 1 + contrast))
    img = ImageEnhance.Color(img).enhance(random.uniform(1 - saturation, 1 + saturation))
    return img

def random_scale(img: Image.Image, boxes, scale_range=(0.8, 1.25)):
    orig_w, orig_h = img.size
    s = random.uniform(*scale_range)
    new_w = max(1, round(orig_w * s))
    new_h = max(1, round(orig_h * s))
    scaled = img.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (orig_w, orig_h), (114, 114, 114))
    left = (orig_w - new_w) // 2
    top = (orig_h - new_h) // 2
    canvas.paste(scaled, (left, top)) # paste the scaled image back onto an original-size canvas

    # Also we have to adjust the bounding boxes
    remapped = []
    for cls_id, xc, yc, w, h in boxes:
        abs_xc = xc * new_w + left
        abs_yc = yc * new_h + top
        abs_w = w * new_w
        abs_h = h * new_h

        xc2 = abs_xc / orig_w
        yc2 = abs_yc / orig_h
        w2 = abs_w / orig_w
        h2 = abs_h / orig_h

        x1 = max(0.0, xc2 - w2 / 2)
        y1 = max(0.0, yc2 - h2 / 2)
        x2 = min(1.0, xc2 + w2 / 2)
        y2 = min(1.0, yc2 + h2 / 2)
        if x2 <= x1 or y2 <= y1:
            continue

        orig_area = w2 * h2
        visible_area = (x2 - x1) * (y2 - y1)
        if orig_area > 0 and visible_area / orig_area < 0.2 : # Added to make sure that if %80 of bounding box is out of the image after scaling. we discard it
            continue

        remapped.append((cls_id, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))

    return canvas, remapped


def apply_clahe(img: Image.Image, clip_limit=2.0, tile_grid_size=(8, 8)):
    arr = np.array(img)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb_eq)


def augment_sample(img: Image.Image, boxes, use_clahe=False):
    img, boxes = horizontal_flip(img, boxes, p=0.5)
    img = limited_color_jitter(img)
    img, boxes = random_scale(img, boxes)
    if use_clahe:
        img = apply_clahe(img)
    return img, boxes