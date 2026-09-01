"""A synthetic Raven's Progressive Matrices generator.

Ninety-six problems is nowhere near enough to train a neural network on. The
standard way out -- and what the research literature does -- is to generate an
unlimited supply of matrices with the same *kinds* of rule, train on those, and
evaluate on the real problems as a genuine out-of-distribution test. The network
never sees a single one of the 96 during training.

Rules generated here mirror the ones the real sets are built from:

  constant      an attribute is fixed along a row
  progression   an attribute steps by a constant amount along a row
  latin         each row is a different permutation of the same three values
  arithmetic    the third cell's count is the sum (or difference) of the first two
  pixel logic   the third panel is the XOR / OR / AND of the first two

Panels are rendered the way the real ones look: black line art on white.
"""
from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image, ImageDraw

SIZE = 184                       # render at the real panel size, then downsample
OUT = 64

# Chosen to cover the shapes the real problem sets actually use.
SHAPES = ["circle", "square", "triangle", "pentagon", "hexagon", "octagon", "star",
          "diamond", "plus", "heart", "pacman", "right_triangle"]
SCALES = [0.34, 0.48, 0.62, 0.78]
ANGLES = [0, 30, 45, 60, 90, 135]
FILLS = [0, 1]
COUNTS = [1, 2, 3, 4]

# 3x3 grid positions inside a 3x3 layout; 2x2 uses the top-left quadrant.
POS_3X3 = list(range(9))
POS_2X2 = [0, 1, 3, 4]


# ------------------------------------------------------------------ drawing

def _polygon(cx, cy, r, n, rot=0.0):
    return [(cx + r * math.cos(rot + 2 * math.pi * i / n - math.pi / 2),
             cy + r * math.sin(rot + 2 * math.pi * i / n - math.pi / 2))
            for i in range(n)]


def _star(cx, cy, r, rot=0.0, points=5):
    pts = []
    for i in range(points * 2):
        rr = r if i % 2 == 0 else r * 0.45
        a = rot + math.pi * i / points - math.pi / 2
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return pts


def draw_shape(d, shape, cx, cy, r, fill, angle, width=3):
    rot = math.radians(angle)
    ink, none = 0, None
    kw = dict(fill=ink if fill else none, outline=ink, width=width)
    if shape == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], **kw)
    elif shape == "square":
        d.polygon(_polygon(cx, cy, r * 1.08, 4, rot + math.pi / 4), **kw)
    elif shape == "triangle":
        d.polygon(_polygon(cx, cy, r, 3, rot), **kw)
    elif shape == "pentagon":
        d.polygon(_polygon(cx, cy, r, 5, rot), **kw)
    elif shape == "hexagon":
        d.polygon(_polygon(cx, cy, r, 6, rot), **kw)
    elif shape == "diamond":
        d.polygon(_polygon(cx, cy, r, 4, rot), **kw)
    elif shape == "star":
        d.polygon(_star(cx, cy, r, rot), **kw)
    elif shape == "octagon":
        d.polygon(_polygon(cx, cy, r, 8, rot + math.pi / 8), **kw)
    elif shape == "right_triangle":
        d.polygon([(cx - r, cy + r * .8), (cx + r, cy + r * .8), (cx - r, cy - r * .8)], **kw)
    elif shape == "heart":
        pts = []
        for i in range(60):
            t = 2 * math.pi * i / 60
            hx = 16 * math.sin(t) ** 3
            hy = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
            ca, sa = math.cos(rot), math.sin(rot)
            pts.append((cx + (hx * ca - hy * sa) * r / 17, cy + (hx * sa + hy * ca) * r / 17))
        d.polygon(pts, **kw)
    elif shape == "pacman":
        start = math.degrees(rot) + 35
        d.pieslice([cx - r, cy - r, cx + r, cy + r], start, start + 290, **kw)
    elif shape == "plus":
        t = r * 0.38
        d.polygon([(cx - t, cy - r), (cx + t, cy - r), (cx + t, cy - t), (cx + r, cy - t),
                   (cx + r, cy + t), (cx + t, cy + t), (cx + t, cy + r), (cx - t, cy + r),
                   (cx - t, cy + t), (cx - r, cy + t), (cx - r, cy - t), (cx - t, cy - t)],
                  **kw)


LAYOUTS = {1: [(0.5, 0.5)],
           2: [(0.32, 0.5), (0.68, 0.5)],
           3: [(0.5, 0.28), (0.3, 0.7), (0.7, 0.7)],
           4: [(0.32, 0.32), (0.68, 0.32), (0.32, 0.68), (0.68, 0.68)]}


def render(attrs, jitter=0.0, width=3) -> np.ndarray:
    """attrs: dict(shape, scale, fill, angle, count) -> uint8 panel, 0=ink.

    `jitter` and `width` are augmentation knobs. The real panels vary slightly
    in stroke weight and centring, so training panels do too -- without it the
    network latches onto the exact pixel geometry of the synthetic renderer and
    transfers worse."""
    im = Image.new("L", (SIZE, SIZE), 255)
    d = ImageDraw.Draw(im)
    n = attrs["count"]
    spots = LAYOUTS[n]
    shrink = {1: 1.0, 2: 0.62, 3: 0.5, 4: 0.46}[n]
    r = attrs["scale"] * SIZE * 0.5 * shrink
    jx = random.uniform(-jitter, jitter) * SIZE
    jy = random.uniform(-jitter, jitter) * SIZE
    for fx, fy in spots:
        draw_shape(d, attrs["shape"], fx * SIZE + jx, fy * SIZE + jy, r,
                   attrs["fill"], attrs["angle"], width)
    return np.array(im.resize((OUT, OUT), Image.BILINEAR))


def blank():
    return np.full((OUT, OUT), 255, dtype=np.uint8)


# ------------------------------------------------------------------ rules

ATTRS = ["shape", "scale", "fill", "angle", "count"]
DOMAIN = {"shape": SHAPES, "scale": SCALES, "fill": FILLS, "angle": ANGLES, "count": COUNTS}


def _row_values(attr, kind, rng):
    """Three values for one attribute across one row, under one rule."""
    dom = DOMAIN[attr]
    if kind == "constant":
        v = rng.choice(dom)
        return [v, v, v]
    if kind == "progression" and attr in ("scale", "count", "angle"):
        idx = rng.randrange(0, max(1, len(dom) - 2))
        step = rng.choice([1, -1])
        idxs = [(idx + step * i) % len(dom) for i in range(3)]
        return [dom[i] for i in idxs]
    if kind == "latin":
        return rng.sample(dom, 3) if len(dom) >= 3 else [rng.choice(dom)] * 3
    v = rng.choice(dom)
    return [v, v, v]


def make_attribute_problem(rng, three=True, n_active=None):
    """A matrix whose rule acts on object attributes (sets B, C and D in spirit).

    Real Raven's problems hold most attributes fixed and vary one or two under a
    rule. An earlier version of this generator gave every attribute its own rule
    at once, which produces visually chaotic matrices that are much harder to
    learn from than the real thing -- see the ablation in the epilogue.
    """
    rows = 3 if three else 2
    cols = 3 if three else 2
    if n_active is None:
        n_active = rng.choice([1, 1, 2])
    active = rng.sample(ATTRS, n_active)

    plan = {}
    for attr in ATTRS:
        if attr not in active:
            plan[attr] = "fixed"          # one value for the whole matrix
            continue
        kinds = ["row_constant", "latin"]
        if attr in ("scale", "count", "angle"):
            kinds += ["progression", "progression"]
        plan[attr] = rng.choice(kinds)

    grid = [[dict() for _ in range(cols)] for _ in range(rows)]
    for attr in ATTRS:
        kind = plan[attr]
        dom = DOMAIN[attr]
        if kind == "fixed":
            v = rng.choice(dom)
            for r in range(rows):
                for c in range(cols):
                    grid[r][c][attr] = v
        elif kind == "latin":
            base = rng.sample(dom, cols) if len(dom) >= cols else [rng.choice(dom)] * cols
            for r in range(rows):
                for c in range(cols):
                    grid[r][c][attr] = base[(c + r) % cols]
        elif kind == "row_constant":
            vals = rng.sample(dom, rows) if len(dom) >= rows else [rng.choice(dom)] * rows
            for r in range(rows):
                for c in range(cols):
                    grid[r][c][attr] = vals[r]
        else:                              # progression along the row
            step = rng.choice([1, -1])
            for r in range(rows):
                start = rng.randrange(len(dom))
                for c in range(cols):
                    grid[r][c][attr] = dom[(start + step * c) % len(dom)]
    return grid, plan


def make_logic_problem(rng, three=True):
    """A matrix whose rule acts on the pixels (set E in spirit)."""
    cols = 3 if three else 2
    rows = 3 if three else 2
    if cols < 3:
        return None
    op = rng.choice(["xor", "or", "and"])
    grid = [[None] * 3 for _ in range(rows)]
    for r in range(rows):
        a = render({"shape": rng.choice(SHAPES), "scale": rng.choice(SCALES[1:]),
                    "fill": 1, "angle": rng.choice(ANGLES), "count": 1})
        b = render({"shape": rng.choice(SHAPES), "scale": rng.choice(SCALES[1:]),
                    "fill": 1, "angle": rng.choice(ANGLES), "count": 1})
        ai, bi = a < 128, b < 128
        ci = {"xor": ai ^ bi, "or": ai | bi, "and": ai & bi}[op]
        grid[r] = [a, b, np.where(ci, 0, 255).astype(np.uint8)]
    return grid, {"logic": op}


def perturb(attrs, rng, n_changes=1):
    out = dict(attrs)
    for attr in rng.sample(ATTRS, n_changes):
        choices = [v for v in DOMAIN[attr] if v != out[attr]]
        if choices:
            out[attr] = rng.choice(choices)
    return out


def make_problem(rng, three=True, augment=True):
    """Returns (context panels, option panels, answer index, kind)."""
    jitter = rng.uniform(0.0, 0.02) if augment else 0.0
    width = rng.choice([2, 3, 3, 4]) if augment else 3
    R = lambda a: render(a, jitter, width)
    use_logic = three and rng.random() < 0.25
    if use_logic:
        made = make_logic_problem(rng, three)
        grid, _ = made
        context = [grid[0][0], grid[0][1], grid[0][2],
                   grid[1][0], grid[1][1], grid[1][2],
                   grid[2][0], grid[2][1]]
        answer = grid[2][2]
        # distractors: other rows' answers and pixel-perturbed variants
        pool = [grid[0][2], grid[1][2]]
        while len(pool) < 7:
            a = render({"shape": rng.choice(SHAPES), "scale": rng.choice(SCALES[1:]),
                        "fill": 1, "angle": rng.choice(ANGLES), "count": 1})
            pool.append(a)
        opts = pool[:7] + [answer]
    else:
        grid, _ = make_attribute_problem(rng, three)
        if three:
            flat = [grid[r][c] for r in range(3) for c in range(3)]
            context = [R(a) for a in flat[:8]]
            truth = flat[8]
            n_opts = 8
        else:
            flat = [grid[r][c] for r in range(2) for c in range(2)]
            context = [R(a) for a in flat[:3]]
            truth = flat[3]
            n_opts = 6
        answer = R(truth)
        seen, opts = {tuple(sorted(truth.items()))}, []
        while len(opts) < n_opts - 1:
            cand = perturb(truth, rng, rng.choice([1, 1, 2]))
            key = tuple(sorted(cand.items()))
            if key in seen:
                continue
            seen.add(key)
            opts.append(R(cand))
        opts.append(answer)

    rng.shuffle(opts)
    ans_idx = next(i for i, o in enumerate(opts) if o is answer)
    return context, opts, ans_idx, ("logic" if use_logic else "attribute")
