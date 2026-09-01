"""Render 'what one API call actually contains' from the real payload.

Walks the exact message structure `03_llm/solver.py::build_input` produces, so
the picture cannot drift from the code. Writes docs/llm_payload.png.
"""
from __future__ import annotations

import base64
import csv
import glob
import io as _io
import os
import re
import sys
import textwrap

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "03_llm"), os.path.join(ROOT, "common")]

import ravens    # noqa: E402
import solver    # noqa: E402

SAMPLE = "Basic Problem D-09"
W = 1180
PAD = 34
INK = (11, 15, 20)
INK2 = (71, 83, 95)
RULE = (185, 205, 229)
PAPER = (255, 255, 255)
BAND = (244, 247, 250)
ACCENT = (235, 104, 52)

F = "/System/Library/Fonts/Supplemental/"
def font(name="Arial.ttf", size=15):
    try:
        return ImageFont.truetype(F + name, size)
    except OSError:
        return ImageFont.load_default()

def mono(size=13):
    for p in ("/System/Library/Fonts/Menlo.ttc", F + "Courier New.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def measured_tokens():
    """Average input tokens per problem, per model, from the runs themselves."""
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "*llm_*_answers.csv"))):
        recs = list(csv.DictReader(open(f)))
        tk = [int(r["InputTokens"]) for r in recs if int(r["InputTokens"]) > 0]
        if not tk:
            continue
        sm = f.replace("_answers.csv", "_summary.txt")
        name = f
        if os.path.exists(sm):
            for line in open(sm):
                if line.startswith("model"):
                    name = line.split(":", 1)[1].strip()
    
        rows.append((name.split("/")[-1], sum(tk) // len(tk)))
    rows.sort(key=lambda r: r[1])
    return rows


def decode(url):
    return Image.open(_io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")


def wrap(draw, text, f, width):
    avg = draw.textlength("x" * 40, font=f) / 40
    return textwrap.wrap(text, max(20, int(width / avg)))


def build():
    problem = next(p for p in ravens.load_all() if p.name == SAMPLE)
    content = solver.build_input(problem)[0]["content"]

    f_h = font("Arial Bold.ttf", 21)
    f_lab = font("Arial Bold.ttf", 13)
    f_b = font("Arial.ttf", 14)
    f_m = mono(12)
    f_s = font("Arial.ttf", 12)

    canvas = Image.new("RGB", (W, 4000), PAPER)
    d = ImageDraw.Draw(canvas)
    y = PAD

    def rule(y):
        d.line([(PAD, y), (W - PAD, y)], fill=RULE, width=1)
        return y + 16

    def label(y, text, colour=INK2):
        d.text((PAD, y), text.upper(), font=f_lab, fill=colour)
        return y + 20

    def para(y, text, f=f_b, colour=INK, x=PAD, width=W - 2 * PAD):
        for line in wrap(d, text, f, width):
            d.text((x, y), line, font=f, fill=colour)
            y += int(f.size * 1.45)
        return y

    d.text((PAD, y), "One API call, one Raven's problem", font=f_h, fill=INK)
    y += 34
    y = para(y, "Every block below is sent in a single request, in this order. "
                "Nothing is pre-solved, cropped or described in words: the model gets "
                "the same pictures a person would.", f_b, INK2)
    y += 10
    y = rule(y)

    # ---- system prompt
    y = label(y, "system prompt  ·  sent once per call")
    box_top = y
    y += 10
    for line in solver.SYSTEM.strip().split("\n"):
        if not line.strip():
            y += 7
            continue
        y = para(y, line.strip(), f_m, INK, PAD + 14, W - 2 * PAD - 28)
    y += 10
    d.rectangle([PAD, box_top, W - PAD, y], outline=RULE, width=1)
    y += 24

    # ---- user content
    y = label(y, f"user content  ·  {len(content)} blocks: "
                 f"{sum(1 for b in content if b['type']=='input_text')} text, "
                 f"{sum(1 for b in content if b['type']=='input_image')} images")
    y += 6

    i = 0
    while i < len(content):
        b = content[i]
        if b["type"] == "input_text":
            txt = b["text"]
            # labels that introduce a run of images get grouped with them below
            if re.fullmatch(r"(Cell|Option) \w+:", txt.strip()):
                run, j = [], i
                while j + 1 < len(content) and content[j]["type"] == "input_text" \
                        and re.fullmatch(r"(Cell|Option) \w+:", content[j]["text"].strip()) \
                        and content[j + 1]["type"] == "input_image":
                    run.append((content[j]["text"].strip().rstrip(":"),
                                decode(content[j + 1]["image_url"])))
                    j += 2
                x, thumb = PAD, 96
                for name, im in run:
                    if x + thumb > W - PAD:
                        x = PAD
                        y += thumb + 30
                    im2 = im.resize((thumb, thumb), Image.LANCZOS)
                    canvas.paste(im2, (x, y + 16))
                    d.rectangle([x, y + 16, x + thumb, y + 16 + thumb], outline=RULE)
                    d.text((x, y), name, font=f_s, fill=INK2)
                    x += thumb + 12
                y += thumb + 34
                i = j
                continue
            y = para(y, txt, f_b, INK)
            y += 8
            i += 1
        else:
            im = decode(b["image_url"])
            wid = 430 if im.width > 400 else 150
            hei = int(im.height * wid / im.width)
            canvas.paste(im.resize((wid, hei), Image.LANCZOS), (PAD, y))
            d.rectangle([PAD, y, PAD + wid, y + hei], outline=RULE)
            ax = PAD + wid + 26
            d.text((ax, y + 2), "the assembled sheet", font=f_lab, fill=INK2)
            d.text((ax, y + 20), f"{im.width} x {im.height} px", font=f_s, fill=INK2)
            ty = y + 52
            d.text((ax, ty), "SAME PAYLOAD, DIFFERENT BILL", font=f_lab, fill=ACCENT)
            ty += 20
            for line in wrap(d, "Every model below received exactly these 17 images. "
                                "What each one counts as an image is entirely its own "
                                "business:", f_s, W - PAD - ax):
                d.text((ax, ty), line, font=f_s, fill=INK2)
                ty += 16
            ty += 8
            for name, tok in measured_tokens():
                d.text((ax, ty), name, font=f_s, fill=INK)
                d.text((W - PAD, ty), f"{tok:,}", font=f_m, fill=INK, anchor="ra")
                ty += 18
            ty += 4
            for line in wrap(d, "input tokens per problem. Token counts are not "
                                "comparable across labs -- only dollars are.",
                             f_s, W - PAD - ax):
                d.text((ax, ty), line, font=f_s, fill=INK2)
                ty += 16
            y = max(y + hei, ty) + 16
            i += 1

    y += 4
    y = rule(y)

    # ---- response
    y = label(y, "the reply is constrained to this json schema", ACCENT)
    box_top = y
    y += 10
    for line in ['{', '  "rule":       string   // the rule, in one sentence',
                 '  "answer":      integer  // 1-8', '  "confidence":  number   // 0-1', '}']:
        d.text((PAD + 14, y), line, font=f_m, fill=INK)
        y += 18
    y += 8
    d.rectangle([PAD, box_top, W - PAD, y], outline=ACCENT, width=1)
    y += 22
    y = para(y, "No tools, no retries on a wrong answer, no few-shot examples, no "
                "conversation. One call per problem, 96 problems, and whatever integer "
                "comes back is the score.", f_b, INK2)

    out = canvas.crop((0, 0, W, y + PAD))
    path = os.path.join(ROOT, "docs", "llm_payload.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.save(path)
    print(f"wrote {path}  {out.size}")
    return path


if __name__ == "__main__":
    build()
