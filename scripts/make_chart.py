"""Render docs/accuracy.png from the parsed results. Called by compare.py."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

# Chart surface and ink -- text never wears a series colour.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e3e2de"

# Categorical slots 1-3, validated for this surface (adjacent CVD dE 9.2,
# normal-vision dE 27.6). Colour encodes the technique family; the bar labels
# carry identity, which is also the relief for aqua's sub-3:1 contrast.
FAMILY_COLOR = {"original": "#2a78d6", "classical": "#eb6834", "llm": "#1baf7a"}
FAMILY_LABEL = {"original": "2017 original", "classical": "Classical AI (no LLM)",
                "llm": "LLM"}


def render(rows, set_rows, set_labels, out_path, n_problems=96, chance=0.135):
    """rows: [(label, correct, kind)] in display order.
       set_rows: [(label, kind, [correct per set])]"""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 8.6), dpi=160,
        gridspec_kw={"height_ratios": [len(rows) * 0.62, 4.4], "hspace": 0.34})
    fig.patch.set_facecolor(SURFACE)

    # ---------------- panel 1: overall accuracy ----------------
    ax1.set_facecolor(SURFACE)
    ys = range(len(rows))[::-1]
    for y, (label, correct, kind) in zip(ys, rows):
        pct = 100 * correct / n_problems
        ax1.barh(y, pct, height=0.6, color=FAMILY_COLOR[kind], zorder=3)
        ax1.text(pct + 1.2, y, f"{correct}/{n_problems}  ({pct:.0f}%)",
                 va="center", ha="left", fontsize=10.5, color=INK, zorder=4)
    ax1.axvline(100 * chance, color=INK_2, lw=1, ls=(0, (4, 3)), zorder=2)
    ax1.text(100 * chance + 0.8, len(rows) - 0.45, "chance", fontsize=9,
             color=INK_2, va="bottom")

    ax1.set_yticks(list(ys))
    ax1.set_yticklabels([r[0] for r in rows], fontsize=10.5, color=INK)
    ax1.set_xlim(0, 118)
    ax1.set_xticks(range(0, 101, 20))
    ax1.set_xticklabels([f"{v}%" for v in range(0, 101, 20)], fontsize=9, color=INK_2)
    ax1.set_title("Accuracy on all 96 Raven's problems", fontsize=13.5,
                  color=INK, loc="left", pad=12)
    _recede(ax1, axis="x")

    # ---------------- panel 2: per problem set ----------------
    ax2.set_facecolor(SURFACE)
    n_series = len(set_rows)
    n_groups = len(set_labels)
    slot = 0.82 / n_series
    width = slot * 0.86                      # leaves a surface gap between bars
    for i, (label, kind, counts) in enumerate(set_rows):
        xs = [g - 0.41 + slot * i + slot / 2 for g in range(n_groups)]
        ax2.bar(xs, counts, width=width, color=FAMILY_COLOR[kind],
                label=label, zorder=3)
    ax2.set_xticks(range(n_groups))
    ax2.set_xticklabels(set_labels, fontsize=10, color=INK)
    ax2.set_ylim(0, 12.8)
    ax2.set_yticks([0, 3, 6, 9, 12])
    ax2.set_yticklabels(["0", "3", "6", "9", "12"], fontsize=9, color=INK_2)
    ax2.set_ylabel("correct of 12", fontsize=10, color=INK_2)
    ax2.set_title("By problem set", fontsize=13.5, color=INK, loc="left", pad=12)
    leg = ax2.legend(frameon=False, fontsize=10, ncol=n_series,
                     loc="upper center", bbox_to_anchor=(0.5, -0.13))
    for t in leg.get_texts():
        t.set_color(INK)
    _recede(ax2, axis="y")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    return out_path


def _recede(ax, axis):
    """Grid and axes stay in the background; no chart junk."""
    for side in ("top", "right", "left" if axis == "x" else "bottom"):
        ax.spines[side].set_visible(False)
    keep = "bottom" if axis == "x" else "left"
    ax.spines[keep].set_color(GRID)
    ax.grid(axis=axis, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, colors=INK_2)


# --------------------------------------------------------------- timeline

def openai_timeline(points, baselines, out_path):
    """points: [(date, label, correct)] for one company, chronological.
       baselines: [(label, correct)] drawn as reference lines."""
    import datetime
    import matplotlib.dates as mdates

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    xs = [datetime.date.fromisoformat(d) for d, _, _ in points]
    ys = [c for _, _, c in points]

    for label, val in baselines:
        ax.axhline(val, color=INK_2, lw=1, ls=(0, (4, 3)), zorder=2)
        ax.text(xs[-1], val + 1.4, label, fontsize=9.5, color=INK_2, ha="right")

    ax.plot(xs, ys, color=FAMILY_COLOR["llm"], lw=2, zorder=3,
            marker="o", markersize=9, markerfacecolor=FAMILY_COLOR["llm"],
            markeredgecolor=SURFACE, markeredgewidth=2)

    # Place each label where it will not collide: a point that sits below its
    # neighbours gets its label underneath, a peak gets it on top, and two
    # models released the same day are forced apart.
    seen = {}
    for i, (x, (d, label, c)) in enumerate(zip(xs, points)):
        nb = [ys[j] for j in (i - 1, i + 1) if 0 <= j < len(ys)]
        above = c >= (sum(nb) / len(nb)) if nb else True
        if seen.get(d):
            above = False
        ha = "center"
        if i == 0:
            ha = "left"
        elif i == len(points) - 1:
            ha = "right"
        seen[d] = seen.get(d, 0) + 1
        ax.annotate(f"{label}\n{c}/96 · {d}", (x, c), textcoords="offset points",
                    xytext=(0, 15 if above else -36), ha=ha, fontsize=9,
                    color=INK, linespacing=1.35)

    ax.set_ylim(20, 108)
    ax.set_yticks([34, 59, 96])
    ax.set_yticklabels(["34", "59", "96"], fontsize=9, color=INK_2)
    ax.set_ylabel("correct out of 96", fontsize=10, color=INK_2)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), fontsize=9, color=INK_2, rotation=0)
    ax.set_title("OpenAI models on the same 96 problems, by release date",
                 fontsize=13.5, color=INK, loc="left", pad=14)
    _recede(ax, axis="y")
    ax.margins(x=0.09)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return out_path


def build_timeline(company="openai"):
    """Convenience entry point: the release-date timeline for one company."""
    import glob
    import model_dates
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dates = model_dates.fetch()
    pts = []
    for path in sorted(glob.glob(os.path.join(root, "results", "*llm_*_summary.txt"))):
        d = {l.split(":", 1)[0].strip(): l.split(":", 1)[1].strip() for l in open(path)}
        model = d.get("model", "")
        lab = model.split("/")[0] if "/" in model else "openai"
        if lab != company:
            continue
        when = model_dates.date_for(model, dates)
        if when:
            pts.append((when, model.split("/")[-1], int(d["correct"])))
    pts.sort()
    return openai_timeline(
        pts, [("hand-written program, no LLM  59", 59), ("2017 student project  34", 34)],
        os.path.join(root, "docs", "openai_timeline.png"))


if __name__ == "__main__":
    print("wrote", build_timeline())
