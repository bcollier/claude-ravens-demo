"""Teaching figures for the README: three worked examples and two failure cases.

Chosen from the results, not by eye:
  easy    every agent solved it, including the 2017 one
  medium  every LLM and the classical agent solved it; the 2017 agent did not
  hard    only 2 of 7 strong LLMs solved it
"""
from __future__ import annotations

import os
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "common"))
import ravens   # noqa: E402

DOCS = os.path.join(ROOT, "docs")
INK, INK2, RULE, ACCENT, GOOD = (11, 15, 20), (71, 83, 95), (185, 205, 229), (235, 104, 52), (27, 175, 122)
F = "/System/Library/Fonts/Supplemental/"


def font(name="Arial.ttf", size=15):
    try:
        return ImageFont.truetype(F + name, size)
    except OSError:
        return ImageFont.load_default()


def sheet(name):
    p = next(x for x in ravens.load_all() if x.name == name)
    return Image.open(p.sheet_path()).convert("RGB"), p


def trim(im, pad=12):
    """Crop the large empty margin off a problem sheet."""
    import numpy as np
    a = np.array(im.convert("L"))
    rows = np.where((a < 245).any(axis=1))[0]
    cols = np.where((a < 245).any(axis=0))[0]
    if not len(rows):
        return im
    return im.crop((max(0, cols[0] - pad), max(0, rows[0] - pad),
                    min(im.width, cols[-1] + pad), min(im.height, rows[-1] + pad)))


def plain(name, out, width=900):
    im = trim(sheet(name)[0])
    im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
    canvas = Image.new("RGB", (im.width + 2, im.height + 2), RULE)
    canvas.paste(im, (1, 1))
    canvas.save(os.path.join(DOCS, out))
    return out, canvas.size


def annotated_hard(out="mistake_two_rules.png", width=980):
    """Challenge B-03: two rules operate at once and the models apply one."""
    im, p = sheet("Challenge Problem B-03")
    im = trim(im)
    im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
    H = im.height + 250
    c = Image.new("RGB", (width, H), (255, 255, 255))
    c.paste(im, (0, 0))
    d = ImageDraw.Draw(c)
    f_h = font("Arial Bold.ttf", 17)
    f_b = font("Arial.ttf", 15)
    f_s = font("Arial.ttf", 13)

    y = im.height + 14
    d.line([(0, y), (width, y)], fill=RULE)
    y += 14
    d.text((0, y), "Two rules are running at the same time", font=f_h, fill=INK)
    y += 26
    for txt, col in [
        ("ACROSS  (A → B):  alternate bands are filled black", INK),
        ("DOWN    (A → C):  two rings are removed, 5 → 3 → 1", ACCENT),
    ]:
        d.text((14, y), txt, font=f_b, fill=col)
        y += 24
    y += 6
    d.text((0, y), "The missing cell needs BOTH: one ring, filled black — "
                   "a plain black square, option 3.", font=f_b, fill=GOOD)
    y += 30
    d.text((0, y), "Five of seven models answered option 1. That is what you get by "
                   "applying only the across rule", font=f_s, fill=INK2)
    y += 20
    d.text((0, y), "to C's three rings and never checking the column. Their own "
                   "explanations describe the fill rule", font=f_s, fill=INK2)
    y += 20
    d.text((0, y), "correctly and never mention the count. The loud rule wins over "
                   "the quiet one.", font=f_s, fill=INK2)
    c.save(os.path.join(DOCS, out))
    return out, c.size


def rule_validation(out="rule_validation.png"):
    """How a candidate rule earns trust: hide a cell you can see, make the rule
    recover it against the real distractors."""
    W, H = 980, 300
    c = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(c)
    f_h = font("Arial Bold.ttf", 14)
    f_c = font("Arial.ttf", 17)
    f_s = font("Arial.ttf", 12)
    f_a = font("Arial Bold.ttf", 13)

    cell, gap, x0, y0 = 54, 6, 8, 56
    names = [["A", "B", "C"], ["D", "E", "F"], ["G", "H", "?"]]
    d.text((x0, 28), "1.  hide a cell you can already see", font=f_h, fill=INK)
    for r in range(3):
        for col in range(3):
            x, y = x0 + col * (cell + gap), y0 + r * (cell + gap)
            hid = (r == 0 and col == 2)
            d.rectangle([x, y, x + cell, y + cell], outline=ACCENT if hid else RULE,
                        width=2 if hid else 1)
            d.text((x + cell / 2, y + cell / 2), names[r][col], font=f_c,
                   fill=ACCENT if hid else INK, anchor="mm")
    d.text((x0, y0 + 3 * cell + 2 * gap + 10), "C is hidden from the rule",
           font=f_s, fill=ACCENT)

    ax = 250
    d.text((ax, 28), "2.  let one candidate rule predict it", font=f_h, fill=INK)
    d.rectangle([ax, y0 + 40, ax + 200, y0 + 110], outline=RULE)
    d.text((ax + 100, y0 + 66), "rule:  C = A over B", font=f_c, fill=INK, anchor="mm")
    d.text((ax + 100, y0 + 90), "one of about 300 candidates", font=f_s,
           fill=INK2, anchor="mm")

    bx = 500
    d.text((bx, 28), "3.  make it compete against the real distractors",
           font=f_h, fill=INK)
    chips = ["C", "1", "2", "3", "4", "5", "6", "7", "8"]
    bars = [.93, .01, .01, .02, .00, .01, .01, .00, .01]
    cw, cg = 44, 6
    for i, (lab, p_) in enumerate(zip(chips, bars)):
        x = bx + i * (cw + cg)
        acc = i == 0
        col = ACCENT if acc else RULE
        d.rectangle([x, y0 + 26, x + cw, y0 + 26 + cw], outline=col, width=2 if acc else 1)
        d.text((x + cw / 2, y0 + 26 + cw / 2), lab, font=f_c,
               fill=ACCENT if acc else INK, anchor="mm")
        h = max(2, p_ * 60)
        d.rectangle([x + 9, y0 + 96 + (60 - h), x + cw - 9, y0 + 156],
                    fill=ACCENT if acc else RULE)
    d.text((bx, y0 + 172), "how much belief the rule puts on each panel",
           font=f_s, fill=INK2)
    d.text((bx, y0 + 192), "trust = 0.93  —  this rule has earned a loud vote",
           font=f_a, fill=GOOD)
    c.save(os.path.join(DOCS, out))
    return out, c.size


def main():
    os.makedirs(DOCS, exist_ok=True)
    made = [
        plain("Basic Problem B-01", "example_easy.png", 720),
        plain("Basic Problem D-09", "example_medium.png", 900),
        plain("Challenge Problem D-08", "mistake_bookkeeping.png", 900),
        annotated_hard(),
        rule_validation(),
        plain("Challenge Problem B-03", "example_hard.png", 760),
    ]
    for name, size in made:
        print(f"wrote docs/{name}  {size}")


if __name__ == "__main__":
    main()
