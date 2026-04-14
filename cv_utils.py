"""OpenCV helpers: smart masks, face detection, feathering."""
from __future__ import annotations
from typing import Tuple
import cv2
import numpy as np
from PIL import Image


def pil_to_cv(img):
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def grabcut_mask(image, rect, iters=5):
    bgr = pil_to_cv(image)
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = rect
    x0, x1 = sorted((max(0, x0), min(w, x1)))
    y0, y1 = sorted((max(0, y0), min(h, y1)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return Image.new("L", (w, h), 0)
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, (x0, y0, x1 - x0, y1 - y0), bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")
    return Image.fromarray(fg)


def threshold_mask(image, thresh=127):
    gray = cv2.cvtColor(pil_to_cv(image), cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(m)


def face_mask(image, pad=0.25):
    bgr = pil_to_cv(image)
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, 1.1, 5)
    mask = np.zeros((h, w), np.uint8)
    for (x, y, fw, fh) in faces:
        px, py = int(fw * pad), int(fh * pad)
        cv2.rectangle(mask, (max(0, x - px), max(0, y - py)),
                      (min(w, x + fw + px), min(h, y + fh + py)), 255, -1)
    return Image.fromarray(mask)


def feather_mask(mask, radius=9):
    m = np.array(mask.convert("L"))
    if radius % 2 == 0:
        radius += 1
    m = cv2.GaussianBlur(m, (radius, radius), 0)
    return Image.fromarray(m)