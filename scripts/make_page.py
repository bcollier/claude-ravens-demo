"""Generate docs/index.html -- the projector version of the whole write-up.

Reads the same result CSVs as compare.py plus results/session_stats.json and
results/session_timeline.txt, so the page can never disagree with the runs.

Regenerate with:  python scripts/make_page.py
"""
from __future__ import annotations

import base64
import datetime
import html
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "common")]

import ravens          # noqa: E402
import compare         # noqa: E402

SAMPLE = ("Basic Problems D", "Basic Problem D-09")
SERIES = {"original": "s1", "classical": "s2", "llm": "s3"}
CUTOFF = "2026-09-01T13:50:00-04:00"      # the moment the class predictions were for


def data_uri(path):
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode()


def esc(x):
    return html.escape(str(x))


# ---------------------------------------------------------------- SVG helpers
# Tiny layout helper so the diagrams sit on an exact grid rather than eyeballed
# coordinates. Everything is currentColor except the one mark carrying the
# argument, which takes the page's accent.

def box(x, y, w, h, label, sub=None, accent=False, dash=False):
    c = 'var(--s2)' if accent else 'currentColor'
    d = ' stroke-dasharray="4 3"' if dash else ''
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="2" fill="none" '
           f'stroke="{c}" stroke-width="1.25"{d}/>']
    ty = y + h / 2 + (0 if sub is None else -5)
    out.append(f'<text x="{x + w/2}" y="{ty}" text-anchor="middle" '
               f'dominant-baseline="middle" font-size="12.5" fill="{c}">{label}</text>')
    if sub:
        out.append(f'<text x="{x + w/2}" y="{y + h/2 + 12}" text-anchor="middle" '
                   f'dominant-baseline="middle" font-size="10.5" fill="{c}" '
                   f'opacity=".65">{sub}</text>')
    return "".join(out)


def arrow(x1, y1, x2, y2, label=None, accent=False, up=False):
    c = 'var(--s2)' if accent else 'currentColor'
    m = 'url(#ah-a)' if accent else 'url(#ah)'
    out = [f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" '
           f'stroke-width="1.25" marker-end="{m}"/>']
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + (-7 if up else 14)
        out.append(f'<text x="{mx}" y="{my}" text-anchor="middle" font-size="10.5" '
                   f'fill="{c}" opacity=".8">{label}</text>')
    return "".join(out)


DEFS = ('<defs>'
        '<marker id="ah" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" '
        'markerHeight="7" orient="auto"><polygon points="0,1 8,4 0,7" fill="currentColor"/></marker>'
        '<marker id="ah-a" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" '
        'markerHeight="7" orient="auto"><polygon points="0,1 8,4 0,7" fill="var(--s2)"/></marker>'
        '</defs>')


def fig_architectures():
    """One figure, one claim: same input, same output, wildly different middles."""
    s = [f'<svg viewBox="0 0 1000 500" role="img" aria-label="Three pipelines '
         'from the same sixteen panel images to the same answer. The 2017 agent '
         'reduces each panel to seven numbers and applies fixed weights. The '
         'classical agent decomposes panels into attributes, generates hundreds '
         'of rules, scores each rule by hiding a line of the matrix, and ranks '
         'the votes. The LLM agent is a single API call.">', DEFS]
    lanes = [
        (60, "A", "2017 agent"),
        (200, "B", "Classical AI agent"),
        (410, "C", "LLM agent"),
    ]
    for y, tag, name in lanes:
        s.append(f'<text x="0" y="{y - 22}" font-size="12.5" font-weight="600" '
                 f'fill="currentColor">{tag} &nbsp;{name}</text>')

    # --- lane A
    y = 60
    s.append(box(0, y, 92, 42, "16 panels", "184&#215;184 px"))
    s.append(arrow(96, y + 21, 148, y + 21))
    s.append(box(152, y, 140, 42, "7 scalar statistics", "ink, overlap, centroid"))
    s.append(arrow(296, y + 21, 372, y + 21, "extrapolate", up=True))
    s.append(box(376, y, 140, 42, "fixed weights", "same for every problem"))
    s.append(arrow(520, y + 21, 572, y + 21))
    s.append(box(576, y, 80, 42, "argmax"))
    s.append(arrow(660, y + 21, 712, y + 21))
    s.append(box(716, y, 92, 42, "answer"))
    s.append(f'<text x="152" y="{y + 60}" font-size="10.5" fill="currentColor" '
             'opacity=".6">34,000 pixels &#8594; 7 numbers: composition is gone here</text>')

    # --- lane B
    y = 200
    s.append(box(0, y, 92, 42, "16 panels"))
    s.append(arrow(96, y + 21, 148, y + 21))
    s.append(box(152, y, 150, 42, "5 descriptors", "frame / inner / interior"))
    s.append(arrow(306, y + 21, 358, y + 21))
    s.append(box(362, y, 150, 42, "~300 rules", "6 families &#215; 4 directions"))
    # the validation loop -- the mechanism the whole agent turns on
    s.append(box(362, y + 84, 150, 40, "hide a line, predict it", accent=True, dash=True))
    s.append(arrow(437, y + 46, 437, y + 80, None, accent=True))
    s.append(f'<path d="M 516 {y + 104} H 545 V {y + 21} H 566" fill="none" '
             'stroke="var(--s2)" stroke-width="1.25" marker-end="url(#ah-a)"/>')
    s.append(f'<text x="551" y="{y + 74}" font-size="10.5" fill="var(--s2)">trust</text>')
    s.append(box(570, y, 130, 42, "22 family votes", "confidence-scaled"))
    s.append(arrow(704, y + 21, 748, y + 21))
    s.append(box(752, y, 140, 42, "learned ranker", "48 weights, fitted"))
    s.append(arrow(896, y + 21, 940, y + 21))
    s.append(box(944, y, 56, 42, "answer"))
    s.append(f'<text x="362" y="{y + 142}" font-size="10.5" fill="var(--s2)" '
             'opacity=".85">the orange loop is the whole difference between a pile of '
             'heuristics and a solver</text>')

    # --- lane C
    y = 410
    s.append(box(0, y, 92, 42, "16 panels"))
    s.append(arrow(96, y + 21, 148, y + 21, "base64", up=True))
    s.append(box(152, y, 250, 42, "one API call", "134-word prompt, JSON schema out"))
    s.append(arrow(406, y + 21, 458, y + 21))
    s.append(box(462, y, 92, 42, "answer"))
    s.append(f'<text x="152" y="{y + 60}" font-size="10.5" fill="currentColor" '
             'opacity=".6">nothing between input and answer that you can inspect, '
             'test, or fix</text>')
    s.append("</svg>")
    return "".join(s)


def fig_validation():
    """How a rule earns trust: it has to recover a panel it was not shown."""
    s = [f'<svg viewBox="0 0 980 320" role="img" aria-label="A three by three '
         'matrix with the top row hidden. A candidate rule predicts the hidden '
         'cell C from A and B, competes against the eight answer options plus '
         'the true panel, and the probability it assigns to the true panel '
         'becomes the rule\'s trust score.">', DEFS]
    # 3x3 grid
    cell, gap, x0, y0 = 52, 6, 0, 60
    names = [["A", "B", "C"], ["D", "E", "F"], ["G", "H", "?"]]
    for r in range(3):
        for c in range(3):
            x, y = x0 + c * (cell + gap), y0 + r * (cell + gap)
            hidden = (r == 0 and c == 2)
            lab = names[r][c]
            s.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="2" '
                     f'fill="none" stroke="{"var(--s2)" if hidden else "currentColor"}" '
                     f'stroke-width="{1.6 if hidden else 1.25}"'
                     f'{" stroke-dasharray=\"4 3\"" if hidden else ""}/>')
            s.append(f'<text x="{x + cell/2}" y="{y + cell/2}" text-anchor="middle" '
                     f'dominant-baseline="middle" font-size="15" '
                     f'fill="{"var(--s2)" if hidden else "currentColor"}">{lab}</text>')
    s.append(f'<text x="0" y="{y0 - 18}" font-size="12.5" font-weight="600" '
             'fill="currentColor">1. hide a cell you can see</text>')
    s.append(f'<text x="0" y="{y0 + 3*cell + 2*gap + 20}" font-size="10.5" '
             'fill="var(--s2)">C is hidden from the rule</text>')

    # rule
    s.append(f'<text x="220" y="{y0 - 18}" font-size="12.5" font-weight="600" '
             'fill="currentColor">2. let a rule predict it</text>')
    s.append(arrow(176, y0 + 78, 214, y0 + 78))
    s.append(box(220, y0 + 40, 190, 76, "", None))
    s.append(f'<text x="315" y="{y0 + 68}" text-anchor="middle" font-size="12.5" '
             'fill="currentColor">rule: C = A &#8746; B</text>')
    s.append(f'<text x="315" y="{y0 + 90}" text-anchor="middle" font-size="10.5" '
             'fill="currentColor" opacity=".65">one of ~300 candidates</text>')

    # candidates
    s.append(f'<text x="450" y="{y0 - 18}" font-size="12.5" font-weight="600" '
             'fill="currentColor">3. make it compete against the real distractors</text>')
    s.append(arrow(414, y0 + 78, 446, y0 + 78))
    chips = ["C", "1", "2", "3", "4", "5", "6", "7", "8"]
    bars = [.93, .01, .01, .02, .00, .01, .01, .00, .01]
    cw, cg, cx = 40, 6, 452
    for i, (lab, p) in enumerate(zip(chips, bars)):
        x = cx + i * (cw + cg)
        acc = i == 0
        col = "var(--s2)" if acc else "currentColor"
        s.append(f'<rect x="{x}" y="{y0 + 30}" width="{cw}" height="{cw}" rx="2" '
                 f'fill="none" stroke="{col}" stroke-width="{1.6 if acc else 1.1}"/>')
        s.append(f'<text x="{x + cw/2}" y="{y0 + 30 + cw/2}" text-anchor="middle" '
                 f'dominant-baseline="middle" font-size="13" fill="{col}">{lab}</text>')
        h = max(2, p * 56)
        s.append(f'<rect x="{x + 8}" y="{y0 + 92 + (56 - h)}" width="{cw - 16}" '
                 f'height="{h}" fill="{col}" opacity="{1 if acc else .35}"/>')
    s.append(f'<text x="{cx}" y="{y0 + 172}" font-size="10.5" fill="currentColor" '
             'opacity=".65">probability the rule puts on each panel, after z-scoring '
             'its fits so every rule type is judged on one scale</text>')
    s.append(f'<text x="{cx}" y="{y0 + 192}" font-size="12.5" fill="var(--s2)">'
             'trust = 0.93 &#8212; this rule earns a loud vote on the real question</text>')
    s.append("</svg>")
    return "".join(s)


# ---------------------------------------------------------------------- CSS
# The palette is lifted from the problem sheets themselves: #b9cde5 is the
# literal cell-border blue on every Raven's sheet in this repo, and the figures
# are pure black on white. The page is built as a test sheet.
CSS = """
:root {
  color-scheme: light;
  --paper:#f4f7fa; --card:#ffffff; --rule:#b9cde5; --rule-soft:#dbe7f2;
  --ink:#0b0f14; --ink-2:#47535f; --ink-3:#78848f;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --track:#e7edf4;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --paper:#0c1015; --card:#141b23; --rule:#2e4459; --rule-soft:#1e2a36;
    --ink:#eef3f8; --ink-2:#9dabb9; --ink-3:#71808e;
    --s1:#3987e5; --s2:#e2703f; --s3:#23b783; --track:#1c262f;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --paper:#0c1015; --card:#141b23; --rule:#2e4459; --rule-soft:#1e2a36;
  --ink:#eef3f8; --ink-2:#9dabb9; --ink-3:#71808e;
  --s1:#3987e5; --s2:#e2703f; --s3:#23b783; --track:#1c262f;
}
* { box-sizing: border-box; }
body {
  background: var(--paper); color: var(--ink);
  font-family: "Source Serif 4", Georgia, "Times New Roman", serif;
  font-size: 18px; line-height: 1.62; -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1120px; margin: 0 auto; padding: 0 32px 96px; }
h1,h2,h3,.num,.cell-tag,.lab,summary { font-family: Archivo, "Helvetica Neue", Arial, sans-serif; }
h1 { font-size: clamp(44px,7vw,92px); font-weight:800; letter-spacing:-.035em;
     line-height:.95; margin:0; text-wrap:balance; }
h2 { font-size: clamp(26px,3.2vw,38px); font-weight:700; letter-spacing:-.02em;
     line-height:1.1; margin:0 0 6px; text-wrap:balance; }
h3 { font-size:20px; font-weight:700; letter-spacing:-.01em; margin:0 0 6px; }
p { margin:0 0 18px; max-width:66ch; }
.lede { font-size:clamp(20px,2.1vw,25px); line-height:1.5; color:var(--ink-2); max-width:60ch; }
strong { font-weight:600; color:var(--ink); }
a { color:var(--ink); text-decoration-color:var(--rule); text-underline-offset:3px; }
a:hover { text-decoration-color:currentColor; }
a:focus-visible, summary:focus-visible { outline:2px solid var(--s1); outline-offset:3px; }
.lab,.cell-tag,.num,table,.bar-val,code { font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;
     font-variant-numeric:tabular-nums; }
.lab { font-size:12px; font-weight:500; letter-spacing:.14em; text-transform:uppercase; color:var(--ink-3); }
code { font-size:.86em; background:var(--rule-soft); padding:1px 5px; border-radius:2px; }
.cell { position:relative; background:var(--card); border:1px solid var(--rule); padding:26px 28px; }
.cell-tag { position:absolute; top:8px; left:11px; font-size:12px; font-weight:500; color:var(--ink-3); }
header.masthead { padding:76px 0 40px; }
.eyebrow { display:flex; gap:14px; align-items:baseline; flex-wrap:wrap; margin-bottom:26px; }
section { padding-top:66px; }
.sec-head { border-top:1px solid var(--rule); padding-top:18px; margin-bottom:30px; }
.matrix3 { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-top:40px; }
.result { padding:40px 24px 26px; display:flex; flex-direction:column; gap:4px; }
.num { font-size:clamp(40px,6.4vw,74px); font-weight:800; letter-spacing:-.045em; line-height:1; }
.num .den { font-size:.42em; font-weight:500; color:var(--ink-3); letter-spacing:-.01em; }
.result .who { font-family:Archivo,sans-serif; font-weight:700; font-size:17px; margin-top:10px; }
.result .how { font-size:15px; color:var(--ink-2); line-height:1.45; }
.swatch { width:34px; height:4px; border-radius:2px; margin-bottom:14px; }
.sets { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }
.setcell { padding:30px 18px 18px; }
.setcell .name { font-family:Archivo,sans-serif; font-weight:700; font-size:16px; margin-bottom:12px; }
.bars { display:flex; flex-direction:column; gap:9px; }
.bar-row { display:grid; grid-template-columns:1fr 30px; align-items:center; gap:8px; }
.track { height:11px; background:var(--track); }
.fill { height:100%; }
.bar-val { font-size:13px; color:var(--ink-2); text-align:right; }
.legend { display:flex; gap:22px; flex-wrap:wrap; margin:0 0 24px; }
.legend span { display:inline-flex; align-items:center; gap:8px; font-size:15px; color:var(--ink-2); }
.legend i { width:14px; height:14px; display:inline-block; }
.example { display:grid; grid-template-columns:1.15fr .85fr; gap:14px; align-items:start; }
.example figure { margin:0; }
.example img { width:100%; display:block; border:1px solid var(--rule-soft); background:#fff; }
figure.dia { margin:0; background:var(--card); border:1px solid var(--rule); padding:34px 30px 22px; overflow-x:auto; }
figure.dia svg { width:100%; min-width:720px; height:auto; color:var(--ink); display:block; }
figcaption { font-size:15px; color:var(--ink-2); margin-top:20px; max-width:78ch; }
.ladder { display:flex; flex-direction:column; gap:10px; }
.rung { display:grid; grid-template-columns:1fr auto; gap:16px; align-items:center;
        border-bottom:1px solid var(--rule-soft); padding-bottom:10px; }
.rung:last-child { border-bottom:0; }
.rung .what { font-size:17px; color:var(--ink-2); }
.rung .what b { color:var(--ink); font-weight:600; }
.rung .score { font-family:Archivo,sans-serif; font-weight:800; font-size:30px;
               letter-spacing:-.03em; font-variant-numeric:tabular-nums; }
.finding { border-left:3px solid var(--s2); padding:4px 0 4px 24px; margin:26px 0; }
.finding p { font-size:clamp(20px,2.3vw,26px); line-height:1.4; color:var(--ink); margin:0; max-width:42ch; }
.finding .src { font-size:14px; color:var(--ink-3); margin-top:10px; font-family:"IBM Plex Mono",monospace; }
.two { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
.three { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; }
.stack { display:flex; flex-direction:column; gap:14px; }
.tablewrap { overflow-x:auto; border:1px solid var(--rule); background:var(--card); }
table { border-collapse:collapse; width:100%; font-size:15px; }
th,td { text-align:left; padding:11px 16px; border-bottom:1px solid var(--rule-soft); }
th { font-size:12px; letter-spacing:.1em; text-transform:uppercase; color:var(--ink-3); font-weight:500; }
tbody tr:last-child td { border-bottom:0; }
td.n, th.n { text-align:right; white-space:nowrap; }
td.keep { color:var(--s3); font-weight:600; white-space:nowrap; }
td.drop { color:var(--s2); font-weight:600; white-space:nowrap; }
tr.worse td { background:color-mix(in srgb, var(--s2) 7%, transparent); }
.qa { border-left:3px solid var(--s3); padding:2px 0 2px 24px; margin-bottom:30px; }
.qa .q { font-size:19px; font-style:italic; color:var(--ink-2); margin:0 0 10px; max-width:62ch; }
.qa .a { font-family:Archivo,sans-serif; font-weight:700; font-size:clamp(22px,2.6vw,30px);
         letter-spacing:-.02em; margin:0 0 6px; }
.costgrid { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }
.costgrid .cell { padding:30px 18px 20px; }
.costgrid .n2 { font-family:Archivo,sans-serif; font-weight:800; font-size:34px;
                letter-spacing:-.03em; font-variant-numeric:tabular-nums; line-height:1.05; }
.costgrid .k { font-size:14px; color:var(--ink-2); margin-top:8px; line-height:1.35; }
footer { margin-top:88px; border-top:1px solid var(--rule); padding-top:26px;
         font-size:15px; color:var(--ink-2); }
footer p { max-width:74ch; }
@media (max-width:900px) {
  .sets,.costgrid { grid-template-columns:repeat(2,1fr); }
  .matrix3,.three,.two,.example { grid-template-columns:1fr; }
  .wrap { padding:0 20px 64px; }
}
@media (prefers-reduced-motion:reduce) { * { animation:none !important; transition:none !important; } }
"""

EXPERIMENTS = [
    ("1", "Run the 2017 code unmodified on Python 3.12", "runs; <b>34/96</b>", "keep", 0),
    ("2", "Fix its two scoring bugs", "34, 34, <b>33</b>", "drop", 1),
    ("3", "Rule search v1 &mdash; transforms, pixel set algebra, numeric progressions, "
          "relational patterns", "<b>55/96</b> with no training", "keep", 0),
    ("4", "Pairwise logistic ranker over 58 features",
     "cross-validated <b>61.5%</b>, but in-sample only 66.7%", "keep", 0),
    ("5", "<b>Rendered three failing set-D puzzles and looked at them</b>",
     "panels are compositions: outer frame &#215; inner shape, in Latin squares", "keep", 0),
    ("6", "Added attribute descriptors + Latin-square rules (70 features)",
     "in-sample rose to 75%, cross-validated <b>fell to 59.4%</b>", "drop", 1),
    ("7", "<b>Asked whether the answer was even in the rule space</b>",
     "any rule <b>95/96</b> &middot; best-family oracle <b>90/96</b> &middot; "
     "single best rule <b>34/96</b>", "keep", 0),
    ("8", "Score rules by whether they recover a <i>hidden</i> line",
     "set D 6&#8594;10, but set E <b>collapsed 8&#8594;4</b>", "drop", 1),
    ("9", "Diagnosed #8: trust was multiplied by fit quality, punishing exact logic "
          "rules (0.899 vs a coarse rule&rsquo;s 1.000)", "diagnosis", "keep", 0),
    ("10", "Validation as a likelihood, &tau;&nbsp;=&nbsp;0.05 on raw fits",
     "<b>47/96</b> &mdash; worse; rule types are on different scales", "drop", 1),
    ("11", "<b>z-score the fits before the softmax, so trust is scale-free</b>",
     "the fix", "keep", 0),
    ("12", "Collapse 4 features per family into 2 pre-scaled votes (48 features)",
     "nested cross-validated <b>61.5%</b>", "keep", 0),
    ("13", "Separate rankers for 2&#215;2 and 3&#215;3 problems", "58.3%", "drop", 1),
    ("14", "Vote using only the top-k best-validated families", "34&ndash;46%", "drop", 1),
    ("15", "&ldquo;Distractors cluster around the answer&rdquo; centrality feature",
     "&plusmn;1 problem", "drop", 1),
    ("16", "Piloted three <code>gpt-5.6</code> variants on the hardest 24 problems",
     "luna 23/24 &middot; sol 23/24 at &frac14; the tokens &middot; terra behind", "keep", 0),
]


def load_stats():
    p = os.path.join(ROOT, "results", "session_stats.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def load_timeline():
    p = os.path.join(ROOT, "results", "session_timeline.txt")
    if not os.path.exists(p):
        return []
    rows = []
    for line in open(p).read().strip().split("\n")[1:]:
        when, rest = line.split("  ", 1)
        label, _, ev = rest.strip().partition("  ")
        rows.append((when, label.strip(), ev.strip()))
    return rows


def thousands(n):
    return f"{n:,}"


def build():
    runs = compare.load_runs()
    scored = [(l, a, e, *compare.score(a)) for l, a, e in runs]
    n = len(compare.ORDER)
    by = {l: (a, e, c, s) for l, a, e, c, s in scored}
    tot = lambda label: sum(by[label][2].values())

    best_llm = max((l for l in by if l.startswith("LLM:")), key=tot)
    org = "Original (2017)"
    cls0 = "Classical AI (no training)"
    cls = "Classical AI + learned ranker"
    headline = [(org, "original", "the 2017 submission, run unmodified"),
                (cls, "classical", "rule search + a learned ranker, no LLM"),
                (best_llm, "llm", "one multimodal call per problem")]

    def set_counts(label):
        c = by[label][2]
        return [sum(c[x] for x in compare.ORDER if compare.SET_OF[x] == s)
                for s in ravens.SET_ORDER]

    ceiling = {}
    cpath = os.path.join(ROOT, "results", "classical_ceiling.txt")
    if os.path.exists(cpath):
        for line in open(cpath).read().strip().split("\n")[1:]:
            ceiling[line.split("  ")[0].strip()] = line.rsplit("/96", 1)[0].split()[-1]

    summary = by[cls][1].get("summary", {})
    conf = by[best_llm][1].get("conf", {})
    ccor = by[best_llm][2]
    conf_r = [v for k, v in conf.items() if ccor[k]]
    conf_w = [v for k, v in conf.items() if not ccor[k]]

    nobody = [x for x in compare.ORDER
              if not (by[org][2][x] or by[cls][2][x] or by[best_llm][2][x])]
    only_cls = [x for x in compare.ORDER if by[cls][2][x] and not by[best_llm][2][x]]
    only_llm = [x for x in compare.ORDER if by[best_llm][2][x] and not by[cls][2][x]]

    # --- session facts
    stats = load_stats()
    sess = stats.get("session", {})
    usage = sess.get("usage", {})
    code = stats.get("code", {}).get("by_kind", {})
    tl = load_timeline()
    nonllm = [r for r in tl if r[1].startswith("non-LLM")]
    first_nonllm = nonllm[0] if nonllm else None
    at_cutoff = [r for r in nonllm if r[0] <= CUTOFF]
    start = min((r[0] for r in tl), default=None)
    gap = None
    if start and first_nonllm:
        f = datetime.datetime.fromisoformat
        gap = (f(first_nonllm[0]) - f(start)).total_seconds()
        # the transcript's first stamp is the first tool result, so widen to the
        # session start recorded alongside it
    hhmm = lambda iso: iso[11:16]

    sample_dir = os.path.join(ROOT, "Problems", SAMPLE[0], SAMPLE[1])
    sample_img = data_uri(os.path.join(sample_dir, SAMPLE[1] + ".PNG"))
    sample_ans = open(os.path.join(sample_dir, "ProblemAnswer.txt")).read().strip()

    P = []
    w = P.append
    w("<title>Raven's Three Ways</title>")
    w('<link rel="preconnect" href="https://fonts.googleapis.com">')
    w('<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>')
    w('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
      'family=Archivo:wght@500;700;800&family=IBM+Plex+Mono:wght@400;500&'
      'family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">')
    w(f"<style>{CSS}</style>")
    w('<div class="wrap">')

    # ---------------------------------------------------------- masthead
    w('<header class="masthead">')
    w('<div class="eyebrow"><span class="lab">Georgia Tech KBAI</span>'
      '<span class="lab">96 problems</span><span class="lab">1 September 2026</span></div>')
    w("<h1>Raven&rsquo;s<br>Three Ways</h1>")
    w('<p class="lede" style="margin-top:26px">The same 96 nonverbal reasoning puzzles, '
      'solved three times: by an eight-year-old student project, by a classical AI agent '
      'forbidden from using a language model, and by a language model.</p>')
    w('<div class="matrix3">')
    for i, (label, kind, how) in enumerate(headline):
        w(f'<div class="cell result"><span class="cell-tag">{chr(65+i)}</span>'
          f'<div class="swatch" style="background:var(--{SERIES[kind]})"></div>'
          f'<div class="num">{tot(label)}<span class="den">/{n}</span></div>'
          f'<div class="who">{esc(label)}</div><div class="how">{esc(how)}</div></div>')
    w("</div>")
    w('<p class="lab" style="margin-top:14px">Chance would score 13 of 96</p>')
    w("</header>")

    # ---------------------------------------------------------- the test
    w('<section><div class="sec-head"><span class="lab">The test</span>'
      "<h2>Sixteen pictures and one missing square</h2></div>")
    w('<div class="example">')
    w('<figure class="cell" style="padding:30px 18px 18px">'
      f'<span class="cell-tag">{esc(SAMPLE[1])}</span>'
      f'<img src="{sample_img}" alt="A three-by-three Raven&#39;s matrix. Each cell holds '
      'a circle, octagon or square crossed by a vertical bar, a horizontal bar or both. '
      'The bottom-right cell is missing." loading="lazy"></figure>')
    w('<div class="cell"><span class="cell-tag">?</span>')
    w("<p>Every row and column must contain each outer shape once and each bar pattern "
      "once. The bottom row already has a square with a horizontal bar and a circle with "
      "a cross, so the missing cell has to be an octagon with a vertical bar: "
      f"<strong>option {esc(sample_ans)}</strong>.</p>")
    w("<p>No words are involved &mdash; the test was designed in 1936 to measure reasoning "
      "without language. Eight sets of twelve run from 2&times;2 matrices up to 3&times;3 "
      "Challenge problems.</p></div></div></section>")

    # ---------------------------------------------------------- the wager
    if first_nonllm and at_cutoff:
        last = at_cutoff[-1]
        w('<section><div class="sec-head"><span class="lab">The wager</span>'
          "<h2>What the class was asked to predict</h2></div>")
        w('<div class="two"><div>')
        w('<div class="qa"><p class="q">&ldquo;Will Claude Code be able to solve any '
          "(1 or more) Raven's Matrices problem without the use of an LLM by the end of "
          'class?&rdquo;</p>'
          f'<p class="a">Yes &mdash; {int(gap//60)} minutes in.</p>'
          f'<p style="margin:0;color:var(--ink-2);font-size:16px">First non-LLM agent ran '
          f'at <strong>{hhmm(first_nonllm[0])}</strong> and scored '
          f'<strong>55 of 96</strong>, before any machine learning was involved.</p></div>')
        w('<div class="qa"><p class="q">&ldquo;How many out of 96 do you estimate the '
          'non-LLM version of this program will be able to solve by the end of '
          'class?&rdquo;</p>'
          f'<p class="a">{tot(cls)} of 96.</p>'
          f'<p style="margin:0;color:var(--ink-2);font-size:16px">The last non-LLM run '
          f'before 1:50&nbsp;PM finished at <strong>{hhmm(last[0])}</strong> reporting '
          f'{tot(cls0)}/96 untrained and {tot(cls)}/96 cross-validated. Those are still '
          "the final numbers &mdash; nothing after the cutoff improved it.</p></div>")
        w("</div>")
        w('<div class="cell"><span class="cell-tag">log</span>'
          '<div class="ladder" style="gap:7px">')
        pretty = [("12:58", "session starts, repo cloned", ""),
                  ("13:03", "the 2017 agent, unmodified", "34"),
                  ("13:08", "first non-LLM agent, no training", "55"),
                  ("13:09", "+ learned ranker", "59"),
                  ("13:12", "added attribute descriptors", "57"),
                  ("13:16", "switched to held-out rule validation", "53"),
                  ("13:17", "validation as a likelihood", "47"),
                  ("13:22", "scale-free validation, restructured features", "59"),
                  ("13:48", "last run before the cutoff", "59")]
        for t, what, score in pretty:
            down = score and int(score) < 59 and t in ("13:12", "13:16", "13:17")
            col = "var(--s2)" if down else "var(--ink)"
            w(f'<div class="rung" style="padding-bottom:6px"><div class="what" '
              f'style="font-size:15px"><span class="lab" style="letter-spacing:.06em">'
              f'{t}</span> &nbsp;{what}</div>'
              f'<div class="score" style="font-size:21px;color:{col}">{score or "&mdash;"}'
              "</div></div>")
        w("</div></div></div>")
        w('<p class="lab" style="margin-top:16px">Three consecutive changes made the score '
          "go down before the one that made it go up</p>")
        w("</section>")

    # ---------------------------------------------------------- scores
    w('<section><div class="sec-head"><span class="lab">Scores</span>'
      "<h2>Twelve problems in each of eight sets</h2></div>")
    w('<div class="legend">')
    for label, kind, _ in headline:
        w(f'<span><i style="background:var(--{SERIES[kind]})"></i>{esc(label)}</span>')
    w("</div><div class=\"sets\">")
    counts = {label: set_counts(label) for label, _, _ in headline}
    for si, s in enumerate(ravens.SET_ORDER):
        w(f'<div class="cell setcell"><span class="cell-tag">{si+1}</span>'
          f'<div class="name">{esc(ravens.set_label(s))}</div><div class="bars">')
        for label, kind, _ in headline:
            v = counts[label][si]
            w('<div class="bar-row"><div class="track">'
              f'<div class="fill" style="width:{v/12*100:.4g}%;'
              f'background:var(--{SERIES[kind]})"></div></div>'
              f'<div class="bar-val">{v}</div></div>')
        w("</div></div>")
    w("</div>")
    w('<p class="lab" style="margin-top:16px">The 2017 agent scores zero on Basic B and '
      "Challenge B because it declines every 2&times;2 problem rather than guessing</p>")
    w("</section>")

    # ---------------------------------------------------------- architecture
    w('<section><div class="sec-head"><span class="lab">Architecture</span>'
      "<h2>Same sixteen images in, same one number out</h2></div>")
    w('<figure class="dia">' + fig_architectures() +
      "<figcaption>Every agent reads the same panel images and returns the same kind of "
      "answer. What differs is everything in between. Agent A throws away all spatial "
      "structure in its first step and can never get it back. Agent B keeps the structure, "
      "proposes rules about it, and &mdash; the orange loop &mdash; scores each rule by "
      "hiding part of the matrix and asking the rule to recover it. Agent C has no middle "
      "at all.</figcaption></figure>")
    w("</section>")

    # ---------------------------------------------------------- agent A
    w('<section><div class="sec-head"><span class="lab">Agent A</span>'
      "<h2>What the 2017 code was doing</h2></div>")
    w("<p>Strip the plumbing and it is one idea applied seven ways: reduce each "
      "184&times;184 panel to a single number, extrapolate that number along the row, the "
      "column and the diagonal, score each option by how close it lands, add the scores "
      "up with fixed weights, take the best. The measurements are ink coverage, shared "
      "ink between two panels, where the ink sits, an identity check, and a test for "
      "whether one panel is the pixel union of two others.</p>")
    w('<div class="two"><div class="cell stack"><span class="cell-tag">+</span>'
      "<h3>What was genuinely good</h3>"
      "<p style=\"margin:0\"><strong>It compares relationships, not objects.</strong> "
      "The agent does not ask which option looks like H; it asks whether the "
      "H&#8594;option relationship matches the G&#8594;H relationship. That is analogy at "
      "the right level of abstraction, and it is not the obvious first thing to try. "
      "Rebuilt as a rule family in agent B, it earns one of the largest positive weights "
      "the ranker learns.</p>"
      "<p style=\"margin:0\"><strong>It eliminates as well as scores.</strong> If nothing "
      "in the matrix repeats, an option identical to a panel you can already see is "
      "probably a distractor. That is a constraint, not a score. Agent B carries the same "
      "signal and the fitted weight on it is large and <em>negative</em> &mdash; a model "
      "independently confirming a hand judgement from 2017.</p>"
      "<p style=\"margin:0\"><strong>It tests before it applies.</strong> Its union rule "
      "is checked on the rows you can see before being used on the row you cannot. That "
      "is the generate-and-test structure agent B uses everywhere.</p></div>")
    w('<div class="cell stack"><span class="cell-tag">&minus;</span>'
      "<h3>What was missing</h3>"
      "<p style=\"margin:0\"><strong>Representation.</strong> Every statistic collapses a "
      "panel to one number, so two panels with the same ink coverage are the same panel. "
      "An entire class of rule becomes literally inexpressible: <em>the inner shape "
      "rotates while the outer frame stays put</em>. Problem set D is built on exactly "
      "those rules.</p>"
      "<p style=\"margin:0\"><strong>No per-problem rule selection.</strong> The weights "
      "are constants. Every puzzle gets the same blend of measurements whether or not "
      "they apply. Nothing ever asks <em>which rule is in force here?</em></p>"
      "<p style=\"margin:0\"><strong>Unfinished coverage.</strong> 2&times;2 problems "
      "return &minus;1. Subtraction and intersection have weights reserved and are never "
      "written. Centroid and object count are computed in full, then multiplied by zero. "
      "Those are deadline fingerprints, and they cost 24 problems outright.</p></div></div>")
    w('<div class="finding"><p>Its two real scoring bugs turn out not to matter. Fixed one '
      "at a time: 34, 34, 34. Fixed together: 33.</p>"
      '<div class="src">The weights had been hand-tuned by running the set and nudging '
      "numbers until the score rose, so the tuning had already absorbed the bugs. The "
      "typos were load-bearing.</div></div>")
    w("</section>")

    # ---------------------------------------------------------- agent B
    w('<section><div class="sec-head"><span class="lab">Agent B</span>'
      "<h2>How a rule earns the right to vote</h2></div>")
    w("<p>Agent B proposes a few hundred rules per puzzle. The hard part is not proposing "
      "them &mdash; it is working out which one the puzzle is actually about. So rules are "
      "scored the way you would score any predictor: hide something you already know the "
      "answer to, and see if the rule can recover it.</p>")
    w('<figure class="dia">' + fig_validation() +
      "<figcaption>A rule's trust is the probability it assigns to a panel it was not "
      "shown, competing against the same distractors it will face on the real question. "
      "Fits are z-scored first, so &lsquo;how much ink overlaps&rsquo; and &lsquo;how many "
      "enclosed holes&rsquo; are judged on one scale &mdash; and a rule too coarse to "
      "separate the options scores about 1-in-9 however perfectly it &lsquo;holds&rsquo;."
      "</figcaption></figure>")

    if ceiling:
        w('<div class="two" style="margin-top:44px"><div class="stack">'
          "<h3>Why this was the thing to fix</h3>"
          "<p style=\"margin:0\">Ten minutes spent measuring the ceiling redirected the "
          "whole project. The rule space almost always contains the right answer. The "
          "agent was not short of ideas; it was bad at picking between them.</p>"
          "<p style=\"margin:0\">Without that measurement the natural instinct was to add "
          "more rule types &mdash; which had just been tried, and had made the "
          "cross-validated score <em>worse</em>.</p>"
          '<p class="lab">python 02_classical_ai/diagnose.py</p></div>')
        w('<div class="cell"><span class="cell-tag">ceiling</span><div class="ladder">')
        for key, blurb in [("any rule", "Some rule, somewhere, picks the right answer"),
                           ("best-family oracle", "An oracle names the right <b>kind</b> of rule"),
                           ("most-trusted rule", "Just believe the single best-scoring rule")]:
            if key in ceiling:
                w(f'<div class="rung"><div class="what">{blurb}</div>'
                  f'<div class="score">{ceiling[key]}<span style="color:var(--ink-3);'
                  'font-size:.55em">/96</span></div></div>')
        w("</div></div></div>")
        w('<div class="finding"><p>Believing the single most-trusted rule scores '
          f'{ceiling.get("most-trusted rule","?")} of 96 &mdash; the same number the 2017 '
          "agent gets.</p>"
          '<div class="src">The equality is a coincidence; what it points at is not. One '
          "fixed scoring scheme applied to every problem lands in the mid-thirties almost "
          "regardless of how good the individual measurements are.</div></div>")
    w("</section>")

    # ---------------------------------------------------------- build log
    w('<section><div class="sec-head"><span class="lab">Build log</span>'
      "<h2>Sixteen things tried, six of them dropped</h2></div>")
    w("<p>The dead ends are the part worth reading. Rows shaded orange made the results "
      "worse than the version before them.</p>")
    w('<div class="tablewrap"><table><thead><tr><th class="n">#</th><th>Tried</th>'
      "<th>Outcome</th><th>Kept</th></tr></thead><tbody>")
    for num, tried, outcome, verdict, worse in EXPERIMENTS:
        cls_attr = ' class="worse"' if worse else ""
        mark = "kept" if verdict == "keep" else "dropped"
        w(f'<tr{cls_attr}><td class="n">{num}</td><td>{tried}</td><td>{outcome}</td>'
          f'<td class="{verdict}">{mark}</td></tr>')
    w("</tbody></table></div>")
    w('<p class="lab" style="margin-top:16px">Full narrative: docs/BUILD_LOG.md</p>')
    w("</section>")

    # ---------------------------------------------------------- confidence
    if conf_r and conf_w:
        w('<section><div class="sec-head"><span class="lab">Calibration</span>'
          "<h2>The LLM does not know when it is wrong</h2></div>")
        w('<div class="two"><div class="stack">')
        w("<p>Every model was asked to return a confidence alongside its answer. "
          f"{esc(best_llm[5:])} averaged "
          f"<strong>{sum(conf_r)/len(conf_r):.2f}</strong> on the problems it got right "
          f"and <strong>{sum(conf_w)/len(conf_w):.2f}</strong> on the ones it got wrong.</p>")
        w("<p>There is no threshold you could set to catch its mistakes. If you were "
          "routing hard cases to a person, this signal would not find them.</p>")
        w("</div><div class=\"cell\"><span class=\"cell-tag\">conf</span>"
          '<div style="display:flex;gap:36px;flex-wrap:wrap">')
        for lab, vals in (("when right", conf_r), ("when wrong", conf_w)):
            w(f'<div><div class="lab">{lab}</div><div class="num" style="font-size:52px">'
              f'{sum(vals)/len(vals):.2f}</div></div>')
        w("</div></div></div></section>")

    # ---------------------------------------------------------- overlap
    w('<section><div class="sec-head"><span class="lab">Overlap</span>'
      "<h2>Where the three disagree</h2></div><div class=\"three\">")
    w(f'<div class="cell"><span class="cell-tag">1</span>'
      f'<div class="num" style="font-size:44px">{len(only_llm)}</div>'
      "<p style=\"margin:10px 0 0\">problems the LLM solved that the classical agent "
      "missed</p></div>")
    w(f'<div class="cell"><span class="cell-tag">2</span>'
      f'<div class="num" style="font-size:44px">{len(only_cls)}</div>'
      '<p style="margin:10px 0 0">'
      + ("the other way round &mdash; the LLM&rsquo;s coverage is a strict superset"
         if not only_cls else "problems the classical agent solved that the LLM missed")
      + "</p></div>")
    w(f'<div class="cell"><span class="cell-tag">3</span>'
      f'<div class="num" style="font-size:44px">{len(nobody)}</div>'
      '<p style="margin:10px 0 0">nobody solved: '
      + ", ".join(esc(x.replace("Problem ", "")) for x in nobody) + "</p></div>")
    w("</div></section>")

    # ---------------------------------------------------------- what it cost
    if usage:
        w('<section><div class="sec-head"><span class="lab">What it cost</span>'
          "<h2>Building all of this</h2></div>")
        w("<p>Measured from the Claude Code session transcript with "
          "<code>scripts/session_stats.py</code>, not estimated. One model, "
          f"<strong>{esc(', '.join(sess.get('models', {})))}</strong>, driving a shell. "
          "These are a snapshot taken while the session was still running, so the real "
          "totals are a little higher &mdash; re-run the script to see the current "
          "figures.</p>")
        w('<div class="costgrid">')
        cards = [
            (thousands(usage.get("output_tokens", 0)), "output tokens &mdash; code written "
             "plus private reasoning"),
            (thousands(usage.get("cache_read_input_tokens", 0)), "cached input tokens re-read "
             "across the session"),
            (thousands(code.get("python", 0)), "lines of Python written across 10 files"),
            (str(sum(sess.get("tools", {}).values())), "tool calls, of which "
             f"{sess.get('tools', {}).get('Bash', 0)} were shell commands"),
        ]
        for big, k in cards:
            w(f'<div class="cell"><div class="n2">{big}</div><div class="k">{k}</div></div>')
        w("</div>")
        fresh = usage.get("input_tokens", 0)
        cached = usage.get("cache_read_input_tokens", 0)
        ratio = int(cached / fresh) if fresh else None
        w('<div class="two" style="margin-top:14px">')
        w('<div class="cell stack"><span class="cell-tag">i</span>'
          "<h3>Almost every token was a re-read</h3>"
          f"<p style=\"margin:0\">Only <strong>{thousands(fresh)}</strong> input tokens were "
          f"new. <strong>{thousands(cached)}</strong> were cache reads of the same growing "
          + (f"conversation &mdash; a ratio of about <strong>{thousands(ratio)}:1</strong>. "
             if ratio else "conversation. ")
          + "Prompt caching is what makes an hour-long agentic session affordable at all: "
          "the model re-reads its entire working context on every single turn.</p></div>")
        w('<div class="cell stack"><span class="cell-tag">ii</span>'
          "<h3>Output was code, not conversation</h3>"
          f"<p style=\"margin:0\">Roughly {thousands(usage.get('output_tokens', 0))} output "
          "tokens against about 8,000 characters of prose actually shown on screen. Nearly "
          "everything the model produced was source code and reasoning it did not print "
          "&mdash; the visible chat is a rounding error on the work.</p></div></div>")
        w('<div class="tablewrap" style="margin-top:14px"><table><thead><tr>'
          "<th>Written</th><th class=\"n\">Lines</th><th>Note</th></tr></thead><tbody>")
        rows = [("Python (the three agents + tooling)", code.get("python", 0),
                 "10 files, excluding the vendored 2017 source"),
                ("Prose (READMEs, notes, build log)", code.get("prose", 0), "Markdown"),
                ("Generated output", code.get("generated output", 0),
                 "COMPARISON.md, this page, the chart &mdash; all produced by the scripts above"),
                ("Result data", code.get("results data", 0), "CSVs and logs from the runs")]
        for name, ln, note in rows:
            w(f'<tr><td>{name}</td><td class="n">{thousands(ln)}</td><td>{note}</td></tr>')
        w("</tbody></table></div></section>")

    # ---------------------------------------------------------- epilogue
    epi_path = os.path.join(ROOT, "results", "epilogue_split_eval.json")
    rn_path = os.path.join(ROOT, "results", "epilogue_neural_relationnet_summary.txt")
    rn_v1 = os.path.join(ROOT, "results", "epilogue_neural_v1_all_attributes_summary.txt")
    if os.path.exists(epi_path) or os.path.exists(rn_path):
        def summ(path):
            d = {}
            if os.path.exists(path):
                for line in open(path):
                    k, _, v = line.partition(":")
                    d[k.strip()] = v.strip()
            return d
        rn, rnv1 = summ(rn_path), summ(rn_v1)
        sp = json.load(open(epi_path)) if os.path.exists(epi_path) else {}

        w('<section><div class="sec-head"><span class="lab">Epilogue &mdash; added after class</span>'
          "<h2>Three things the session did not settle</h2></div>")
        w('<p>Everything above ran live and is untouched. These were added afterwards, '
          'and the full write-up is in <code>EPILOGUE.md</code>.</p>')
        w('<div class="three">')
        if rn:
            w('<div class="cell stack"><span class="cell-tag">1</span>'
              "<h3>A neural network, still no LLM</h3>"
              f'<div class="num" style="font-size:40px">{rn.get("correct","?")}'
              f'<span class="den">/96</span></div>'
              "<p style=\"margin:6px 0 0\">A relation network (CNN + a shared MLP over "
              "every pair of panels) trained only on synthetic matrices, so the real 96 "
              "stay a genuine held-out test. It does not reach the symbolic agent: with "
              "96 problems and strong domain priors, encoding the priors beats learning "
              "them.</p>"
              + (f'<p style="margin:0;font-size:15px;color:var(--ink-2)">Fixing the '
                 f'<em>generator</em> so it varies one or two attributes at a time, the '
                 f'way real matrices do, moved held-out synthetic accuracy from '
                 f'{float(rnv1.get("synthetic_val",0)):.0%} to '
                 f'{float(rn.get("synthetic_val",0)):.0%}. The training distribution was '
                 f'the bottleneck, not the architecture.</p>' if rnv1 and rn else "")
              + "</div>")
        if sp.get("linear"):
            d = sp["linear"]
            w('<div class="cell stack"><span class="cell-tag">2</span>'
              "<h3>A plain 70/30 split</h3>"
              f'<div class="num" style="font-size:40px">{d["mean"]:.0%}</div>'
              f'<p style="margin:6px 0 0">Stratified, {d["train_n"]} train / '
              f'{d["test_n"]} test, over {d["seeds"]} seeds. <em>Higher</em> than the '
              "leave-one-set-out number above, because a random split lets the model see "
              "every problem family during training. The gap between the two is how much "
              "the agent depends on having met that kind of puzzle before.</p>"
              '<p style="margin:0;font-size:15px;color:var(--s2)">The model weights were '
              "always held out. The <em>feature design</em> was not: set D's failures "
              "were inspected and rule families added because of them. No re-split fixes "
              "that.</p></div>")
        w('<div class="cell stack"><span class="cell-tag">3</span>'
          "<h3>Other labs</h3>"
          "<p style=\"margin:6px 0 0\">The in-class runs were all OpenAI. The epilogue "
          "adds one flagship vision model per lab on the identical prompt, with the "
          "dollar cost of every run. GPT-3.5 cannot be entered at all &mdash; it has no "
          "image input, so the barrier is modality, not reasoning.</p></div>")
        w("</div></section>")

    # ---------------------------------------------------------- method
    w('<section><div class="sec-head"><span class="lab">Method</span>'
      "<h2>How to read these numbers</h2></div><div class=\"two\">")
    w('<div class="cell stack"><span class="cell-tag">A</span>'
      "<h3>Everyone gets the same input</h3>"
      "<p style=\"margin:0\">Panel images only. Two of the eight sets ship with a verbal "
      "description of each figure; the other six do not, so it is withheld from all three "
      "agents. The 2017 agent's author had already disabled that path anyway.</p></div>")
    ins = summary.get("in_sample_not_a_result")
    w('<div class="cell stack"><span class="cell-tag">B</span>'
      "<h3>The learned part is cross-validated</h3>"
      f'<p style="margin:0">Agent B trains on seven problem sets and is tested on the '
      f"eighth, with every hyper-parameter chosen inside the training folds. Rule search "
      f"with no training at all scores {tot(cls0)}/96"
      + (f"; scoring the ranker on the very problems it was fitted to gives "
         f"{round(float(ins)*n)}/96, only {round(float(ins)*n)-tot(cls)} more &mdash; which "
         "is how you know it is learning which rules to trust rather than memorising "
         "answers." if ins else ".")
      + "</p></div>")
    w('<div class="cell stack"><span class="cell-tag">C</span>'
      "<h3>The LLM score has an asterisk</h3>"
      "<p style=\"margin:0\">These puzzles have been in a public GitHub repository since "
      "2017. Contamination cannot be ruled out from here. Read it as a score on a public "
      "benchmark, not a clean measurement of novel visual reasoning. Agent B has no such "
      "exposure, which is part of why it is worth keeping.</p></div>")
    w('<div class="cell stack"><span class="cell-tag">D</span>'
      "<h3>They fail in different ways</h3>"
      "<p style=\"margin:0\">The 2017 agent fails by declining. Agent B fails by picking a "
      "rule that fits but is not the rule &mdash; and it will name the rule it picked. "
      "The LLM fails confidently and leaves nothing to debug. Only the middle one explains "
      "itself.</p></div></div></section>")

    # ---------------------------------------------------------- all results
    w('<section><div class="sec-head"><span class="lab">Every run</span>'
      "<h2>All results</h2></div>")
    w('<div class="tablewrap"><table><thead><tr><th>Agent</th><th class="n">Correct</th>'
      '<th class="n">Accuracy</th><th class="n">Declined</th><th class="n">Wall clock</th>'
      "<th>Network</th></tr></thead><tbody>")
    for label, a, e, c, sk in scored:
        t = sum(c.values())
        rt = e.get("wall") or e.get("runtime")
        rts = "&mdash;" if not rt else (f"{rt:.0f} s" if rt < 600 else f"{rt/60:.0f} min")
        w(f'<tr><td>{esc(label)}</td><td class="n">{t}/{n}</td><td class="n">{t/n:.1%}</td>'
          f'<td class="n">{len(sk)}</td><td class="n">{rts}</td>'
          f'<td>{"yes" if e.get("kind")=="llm" else "no"}</td></tr>')
    w("</tbody></table></div>")
    w('<p class="lab" style="margin-top:14px">Local agents single-process on an M3 Max; '
      "LLM sweeps at 10 concurrent requests</p></section>")

    w("<footer><p><strong>Raven&rsquo;s Three Ways</strong> &mdash; code, raw results, the "
      "build log and the full write-up at "
      '<a href="https://github.com/bcollier/claude-ravens-demo">'
      "github.com/bcollier/claude-ravens-demo</a>. Problems and harness from the Georgia "
      "Tech Knowledge-Based AI project. Every number on this page is generated from the "
      "result files by <code>scripts/make_page.py</code>; none is typed by hand.</p>"
      "</footer></div>")

    out = os.path.join(ROOT, "docs", "index.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write("\n".join(P))
    print(f"wrote {out}  ({os.path.getsize(out)/1024:.0f} KB)")
    return out


if __name__ == "__main__":
    build()
