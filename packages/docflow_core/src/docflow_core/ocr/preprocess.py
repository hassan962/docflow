"""Page clean-up before OCR: grayscale, denoise, deskew.

Each step is a pure function (image in, image out) so it can be tested on its own with the
synthetic noisy invoices, and the pipeline is just their composition.

Deskew uses the projection-profile method, which needs only NumPy: text lines are
horizontal stripes of ink, so when the page is straight the per-row ink counts swing
sharply between "line" and "gap". Rotating by each candidate angle and keeping the one with
the sharpest profile recovers the skew. A coarse pass then a fine pass keeps it fast.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter, ImageOps

WHITE = 255


def to_grayscale(img: Image.Image) -> Image.Image:
    return ImageOps.grayscale(img)


def denoise(img: Image.Image, size: int = 3) -> Image.Image:
    """Median filter: removes salt-and-pepper speckle while keeping character edges."""
    return img.filter(ImageFilter.MedianFilter(size))


def _profile_score(binary: Image.Image, angle: float) -> float:
    rotated = binary.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)
    rows = np.asarray(rotated, dtype=np.float64).sum(axis=1)
    return float(np.sum(np.diff(rows) ** 2))


def estimate_skew(img: Image.Image, max_angle: float = 5.0) -> float:
    """Angle in degrees to rotate the page by (counter-clockwise) to straighten it."""
    gray = to_grayscale(img)
    scale = 1000 / max(gray.size)
    if scale < 1:
        gray = gray.resize((int(gray.width * scale), int(gray.height * scale)))
    # Ink = 1, paper = 0. Otsu-like threshold is overkill here; the mean works for documents.
    arr = np.asarray(gray)
    binary = Image.fromarray(((arr < arr.mean() * 0.8) * 255).astype(np.uint8))

    def best(candidates: np.ndarray) -> float:
        scores = [_profile_score(binary, float(a)) for a in candidates]
        return float(candidates[int(np.argmax(scores))])

    coarse = best(np.arange(-max_angle, max_angle + 0.01, 0.5))
    return round(best(np.arange(coarse - 0.5, coarse + 0.51, 0.1)), 2)


def deskew(img: Image.Image, max_angle: float = 5.0) -> Image.Image:
    angle = estimate_skew(img, max_angle)
    if abs(angle) < 0.1:
        return img
    # Same canvas size, so page dimensions (and therefore box coordinates) stay stable.
    return img.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=WHITE)


def _long_runs(ink: np.ndarray, length: int, axis: int) -> np.ndarray:
    """Pixels that belong to an unbroken run of ink at least ``length`` long along ``axis``."""
    if ink.shape[axis] < length:
        return np.zeros_like(ink)
    a = np.moveaxis(ink, axis, -1).astype(np.int32)
    csum = np.concatenate([np.zeros((*a.shape[:-1], 1), np.int32), np.cumsum(a, axis=-1)], -1)
    full = (csum[..., length:] - csum[..., :-length]) == length  # window start is all ink
    # Spread each full window back over the pixels it covers.
    starts = np.concatenate([np.zeros((*a.shape[:-1], 1), np.int32), np.cumsum(full, -1)], -1)
    n = a.shape[-1]
    idx = np.arange(n)
    covered = (
        starts[..., np.minimum(idx + 1, n - length + 1)]
        - starts[..., np.maximum(idx - length + 1, 0)]
    )
    return np.moveaxis(covered > 0, -1, axis)


def _thin(mask: np.ndarray, thickness: int, axis: int) -> np.ndarray:
    """Keep mask pixels whose neighbours ``thickness`` away (across the line) are not mask."""
    before = np.roll(mask, thickness, axis=axis)
    after = np.roll(mask, -thickness, axis=axis)
    thin: np.ndarray = mask & ~before & ~after
    return thin


def remove_lines(img: Image.Image, min_fraction: float = 0.08, thickness: int = 6) -> Image.Image:
    """Erase thin ruling lines (table borders, boxes) so OCR layout analysis is not misled.

    A pixel is erased when it lies on a straight run of ink longer than ``min_fraction`` of
    the page and the line it belongs to is at most about ``thickness`` pixels thick. Solid
    areas such as a black table-header bar are thick, so they and the white text inside
    them are kept. Pixels are only whitened, never moved, so every box found on the cleaned
    image is valid on the original too.
    """
    gray = to_grayscale(img)
    arr = np.asarray(gray).copy()
    ink = arr < 128
    horizontal = _thin(_long_runs(ink, int(arr.shape[1] * min_fraction), axis=1), thickness, 0)
    vertical = _thin(_long_runs(ink, int(arr.shape[0] * min_fraction), axis=0), thickness, 1)
    arr[horizontal | vertical] = WHITE
    return Image.fromarray(arr)


def preprocess(img: Image.Image) -> Image.Image:
    """The page image that is stored and shown to reviewers."""
    return deskew(denoise(to_grayscale(img)))


def ocr_input(page: Image.Image) -> Image.Image:
    """What the OCR engine reads: the stored page with ruling lines erased."""
    return remove_lines(page)
