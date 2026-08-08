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
    s = random.uniform(*scale_range)
    new_w = max(1, round(img.width * s))
    new_h = max(1, round(img.height * s))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    return img, boxes


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