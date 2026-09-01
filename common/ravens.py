"""Shared loader for the Raven's Progressive Matrices problem sets.

One place that knows how the 2017 KBAI folder layout works, so all three
solvers see exactly the same 96 problems and the same ground truth.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBLEMS_DIR = os.path.join(ROOT, "Problems")

SET_ORDER = [
    "Basic Problems B", "Basic Problems C", "Basic Problems D", "Basic Problems E",
    "Challenge Problems B", "Challenge Problems C", "Challenge Problems D", "Challenge Problems E",
]

# Matrix cells (givens) and answer options, by problem type.
GIVENS = {"2x2": ["A", "B", "C"], "3x3": list("ABCDEFGH")}
N_CHOICES = {"2x2": 6, "3x3": 8}


@dataclass
class Problem:
    name: str
    set_name: str
    problem_type: str          # "2x2" or "3x3"
    has_verbal: bool
    folder: str
    answer: int                # ground truth, 1-based
    # verbal representation: {figure_name: {object_name: {attr: value}}}
    verbal: dict = field(default_factory=dict)
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def givens(self) -> list[str]:
        return GIVENS[self.problem_type]

    @property
    def choices(self) -> list[str]:
        return [str(i) for i in range(1, N_CHOICES[self.problem_type] + 1)]

    @property
    def n_choices(self) -> int:
        return N_CHOICES[self.problem_type]

    def path(self, figure: str) -> str:
        return os.path.join(self.folder, figure + ".png")

    def sheet_path(self) -> str:
        """The full problem sheet image (all panels + options in one picture)."""
        return os.path.join(self.folder, self.name + ".PNG")

    def image(self, figure: str) -> np.ndarray:
        """Binary ink mask for a figure: True where there is ink (dark pixel)."""
        if figure not in self._cache:
            im = Image.open(self.path(figure)).convert("L")
            self._cache[figure] = np.array(im) < 128
        return self._cache[figure]


def _parse_verbal(lines: list[str]) -> dict:
    figures: dict = {}
    current_fig = current_obj = None
    for line in lines:
        if not line.strip():
            continue
        if not line.startswith("\t"):
            current_fig = line.strip()
            figures[current_fig] = {}
        elif not line.startswith("\t\t"):
            current_obj = line.strip()
            figures[current_fig][current_obj] = {}
        else:
            key, _, value = line.strip().partition(":")
            figures[current_fig][current_obj][key] = value
    return figures


def load_problem(set_name: str, problem_name: str) -> Problem:
    folder = os.path.join(PROBLEMS_DIR, set_name, problem_name)
    with open(os.path.join(folder, "ProblemData.txt")) as fh:
        raw = fh.read().split("\n")
    problem_type = raw[0].strip()
    has_visual = raw[1].strip() == "true"
    has_verbal = raw[2].strip() == "true"
    assert has_visual, problem_name
    with open(os.path.join(folder, "ProblemAnswer.txt")) as fh:
        answer = int(fh.read().strip())
    return Problem(
        name=problem_name,
        set_name=set_name,
        problem_type=problem_type,
        has_verbal=has_verbal,
        folder=folder,
        answer=answer,
        verbal=_parse_verbal(raw[3:]) if has_verbal else {},
    )


def load_all(sets: list[str] | None = None) -> list[Problem]:
    problems = []
    for set_name in (sets or SET_ORDER):
        list_file = os.path.join(PROBLEMS_DIR, set_name, "ProblemList.txt")
        with open(list_file) as fh:
            names = [ln.strip() for ln in fh if ln.strip()]
        for name in names:
            problems.append(load_problem(set_name, name))
    return problems


def set_label(set_name: str) -> str:
    """'Basic Problems D' -> 'Basic D'"""
    return re.sub(r"\s*Problems\s*", " ", set_name).strip()
