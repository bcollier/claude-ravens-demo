"""Binary-image primitives for the classical (non-LLM) solver.

Every figure is an ink mask: a boolean HxW array, True where the panel has ink.
Nothing here is deep-learning; it is classical computer vision + set algebra.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

# ---------------------------------------------------------------- similarity

def ink(a: np.ndarray) -> float:
    """Fraction of the panel covered in ink."""
    return float(a.mean())


def tanimoto(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of the ink pixels. 1.0 == identical."""
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    if union == 0:
        return 1.0          # two blank panels are identical
    return inter / union


def agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Plain per-pixel agreement (dominated by shared white space)."""
    return float((a == b).mean())


def sim(a: np.ndarray, b: np.ndarray) -> float:
    """Blended similarity: IoU is discriminative, agreement is stable."""
    return 0.75 * tanimoto(a, b) + 0.25 * agreement(a, b)


# ---------------------------------------------------------------- transforms

def _rot(k):
    return lambda a: np.rot90(a, k)


TRANSFORMS = {
    "identity":   lambda a: a,
    "flip_h":     np.fliplr,
    "flip_v":     np.flipud,
    "rot90":      _rot(1),
    "rot180":     _rot(2),
    "rot270":     _rot(3),
    "transpose":  lambda a: a.T,
    "anti_transpose": lambda a: np.rot90(a, 2).T,
}

# ---------------------------------------------------------------- set algebra

LOGIC_OPS = {
    "and":   lambda a, b: a & b,
    "or":    lambda a, b: a | b,
    "xor":   lambda a, b: a ^ b,
    "a_not_b": lambda a, b: a & ~b,
    "b_not_a": lambda a, b: b & ~a,
    "nand_ink": lambda a, b: ~(a & b) & (a | b),   # symmetric difference of the union
}

# ---------------------------------------------------------------- shape norm

def bbox(a: np.ndarray):
    rows = np.flatnonzero(a.any(axis=1))
    cols = np.flatnonzero(a.any(axis=0))
    if rows.size == 0:
        return None
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1


def normalize_shape(a: np.ndarray, size: int = 48) -> np.ndarray:
    """Crop to the ink bounding box and rescale, so translation/scale drop out."""
    box = bbox(a)
    if box is None:
        return np.zeros((size, size), dtype=bool)
    r0, r1, c0, c1 = box
    crop = a[r0:r1, c0:c1]
    zoom = (size / crop.shape[0], size / crop.shape[1])
    out = ndimage.zoom(crop.astype(np.float32), zoom, order=1)
    out = out[:size, :size]
    if out.shape != (size, size):     # pad any rounding shortfall
        pad = np.zeros((size, size), dtype=np.float32)
        pad[:out.shape[0], :out.shape[1]] = out
        out = pad
    return out > 0.5


# ---------------------------------------------------------------- scalar feats

_STRUCT8 = np.ones((3, 3), dtype=bool)


def n_objects(a: np.ndarray) -> int:
    """Connected components of ink (8-connected) -- a proxy for 'how many shapes'."""
    return int(ndimage.label(a, structure=_STRUCT8)[1])


def n_holes(a: np.ndarray) -> int:
    """Enclosed background regions -- distinguishes outlines from filled shapes."""
    bg = ~a
    lab, n = ndimage.label(bg)
    if n == 0:
        return 0
    border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    return sum(1 for i in range(1, n + 1) if i not in border)


def centroid(a: np.ndarray):
    idx = np.nonzero(a)
    if idx[0].size == 0:
        return (a.shape[0] / 2, a.shape[1] / 2)
    return (float(idx[0].mean()), float(idx[1].mean()))


def bbox_area(a: np.ndarray) -> float:
    box = bbox(a)
    if box is None:
        return 0.0
    r0, r1, c0, c1 = box
    return (r1 - r0) * (c1 - c0) / a.size


def symmetry_h(a: np.ndarray) -> float:
    return tanimoto(a, np.fliplr(a))


def symmetry_v(a: np.ndarray) -> float:
    return tanimoto(a, np.flipud(a))


def scalar_features(a: np.ndarray) -> dict:
    """Per-panel scalars used for arithmetic-progression hypotheses."""
    cy, cx = centroid(a)
    return {
        "ink": ink(a),
        "objects": float(n_objects(a)),
        "holes": float(n_holes(a)),
        "bbox_area": bbox_area(a),
        "cy": cy / a.shape[0],
        "cx": cx / a.shape[1],
        "sym_h": symmetry_h(a),
        "sym_v": symmetry_v(a),
    }


SCALAR_NAMES = ["ink", "objects", "holes", "bbox_area", "cy", "cx", "sym_h", "sym_v"]


# ---------------------------------------------------------------- alignment

def recenter(a: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Translate `a` so its ink centroid sits on `ref`'s -- kills small offsets."""
    ay, ax = centroid(a)
    ry, rx = centroid(ref)
    dy, dx = int(round(ry - ay)), int(round(rx - ax))
    if dy == 0 and dx == 0:
        return a
    return np.roll(np.roll(a, dy, axis=0), dx, axis=1)


def sim_centered(a: np.ndarray, b: np.ndarray) -> float:
    return sim(recenter(a, b), b)


# ---------------------------------------------------------------- descriptors
# A panel is usually a composition of attributes (an outer frame, an inner
# shape, some interior detail). These pull those apart so a rule can be tested
# against one attribute at a time.

def components(a: np.ndarray):
    """Ink components, largest bounding box first."""
    lab, n = ndimage.label(a, structure=_STRUCT8)
    if n == 0:
        return []
    out = []
    for i in range(1, n + 1):
        m = lab == i
        box = bbox(m)
        out.append(((box[1] - box[0]) * (box[3] - box[2]), m))
    out.sort(key=lambda t: -t[0])
    return [m for _, m in out]


def silhouette(a: np.ndarray) -> np.ndarray:
    """Solid outer blob -- the panel's overall shape, ignoring interior detail."""
    return ndimage.binary_fill_holes(a)


def core(a: np.ndarray, erode: int = 6) -> np.ndarray:
    """Ink strictly inside the silhouette -- the interior detail."""
    inside = ndimage.binary_erosion(silhouette(a), iterations=erode)
    return a & inside


def _first_or_empty(comps, idx, shape):
    if not comps:
        return np.zeros(shape, dtype=bool)
    return comps[idx]


DESCRIPTORS = {
    "whole": lambda a: normalize_shape(a),
    "sil":   lambda a: normalize_shape(silhouette(a)),
    "outer": lambda a: normalize_shape(_first_or_empty(components(a), 0, a.shape)),
    "inner": lambda a: normalize_shape(_first_or_empty(components(a), -1, a.shape)),
    "core":  lambda a: normalize_shape(core(a)),
}
