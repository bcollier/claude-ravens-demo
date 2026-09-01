"""Generate-and-test analogy engine with held-out rule validation.

For each problem we enumerate a few hundred candidate *rules* about how a line
of the matrix (row, column, wrapped diagonal) progresses. The hard part is not
generating rules -- it is deciding which one is in force. We do that the way
you would evaluate any predictor: hide a line whose answer we already know,
make the rule predict it against the real distractor set, and see whether it
picks the right panel.

    vconf(rule) = how often, and how decisively, the rule recovers a panel
                  we can already see, competing against the answer options

That single change turns a pile of pixel heuristics into a rule-selection
problem with an honest internal scoring signal. No LLM, no neural network.
"""
from __future__ import annotations

import itertools

import numpy as np

import imageops as io

# --------------------------------------------------------------- geometry
# Complete lines have every cell visible and are used to score rules.
# The prefix is the start of the line the answer has to finish.

DIRECTIONS_3X3 = {
    "row":   ([("A", "B", "C"), ("D", "E", "F")], ("G", "H")),
    "col":   ([("A", "D", "G"), ("B", "E", "H")], ("C", "F")),
    # wrapped diagonals -- the Latin-square patterns common in set D
    "diag":  ([("D", "H", "C"), ("G", "B", "F")], ("A", "E")),
    "adiag": ([("A", "F", "H"), ("C", "E", "G")], ("B", "D")),
}

DIRECTIONS_2X2 = {
    "row": ([("A", "B")], ("C",)),
    "col": ([("A", "C")], ("B",)),
}


def directions(problem):
    return DIRECTIONS_3X3 if problem.problem_type == "3x3" else DIRECTIONS_2X2


# --------------------------------------------------------------- panel cache

class Panels:
    """Lazily computed views of every figure in one problem."""

    def __init__(self, problem):
        self.p = problem
        self.names = problem.givens + problem.choices
        self.raw = {n: problem.image(n) for n in self.names}
        self._desc = {}
        self._scalars = {}

    def desc(self, kind, n):
        key = (kind, n)
        if key not in self._desc:
            self._desc[key] = io.DESCRIPTORS[kind](self.raw[n])
        return self._desc[key]

    def scalars(self, n):
        if n not in self._scalars:
            self._scalars[n] = io.scalar_features(self.raw[n])
        return self._scalars[n]

    def scalar_scale(self, key):
        """Typical spread of a scalar in this problem -- the unit of 'close'."""
        vals = [self.scalars(n)[key] for n in self.names]
        floor = {"objects": 0.5, "holes": 0.5}.get(key, 0.02)
        return max(float(np.std(vals)), floor)


def _closeness(err, scale):
    return max(0.0, 1.0 - abs(err) / max(scale, 1e-6))


# --------------------------------------------------------------- rule makers
# A rule maker turns the evidence lines into {name: fit_fn}, where
#     fit_fn(prefix, candidates) -> array of fits in [0, 1].
# Rules that need no evidence (transforms, set algebra) ignore it.

def transform_rules(pan, evidence, kind=None):
    """T carries each cell of the line to the next. Identity == 'unchanged'."""
    if kind is None:
        get, measures = (lambda n: pan.raw[n]), (("raw", io.sim), ("centered", io.sim_centered))
        tset = io.TRANSFORMS
    else:
        get, measures = (lambda n: pan.desc(kind, n)), (("shape", io.tanimoto),)
        tset = ATTR_TRANSFORMS
    rules = {}
    for tname, T in tset.items():
        for aname, measure in measures:
            def fit(prefix, cands, T=T, measure=measure, get=get):
                pred = T(get(prefix[-1]))
                return np.array([measure(pred, get(c)) for c in cands])
            rules[f"{tname}/{aname}"] = fit
    return rules


def logic_rules(pan, evidence):
    """Pixel set algebra across a line of three: op(x1, x2) == x3."""
    rules = {}
    for opname, op in io.LOGIC_OPS.items():
        def fit(prefix, cands, op=op):
            if len(prefix) < 2:
                return np.zeros(len(cands))
            pred = op(pan.raw[prefix[0]], pan.raw[prefix[1]])
            return np.array([io.sim(pred, pan.raw[c]) for c in cands])
        rules[opname] = fit
    return rules


def numeric_rules(pan, evidence):
    """A measurable property progresses predictably along the line."""
    rules = {}
    for key in io.SCALAR_NAMES:
        scale = pan.scalar_scale(key)
        v = lambda n, key=key: pan.scalars(n)[key]

        def const(prefix, cands, v=v, scale=scale):
            return np.array([_closeness(v(c) - v(prefix[-1]), 2 * scale) for c in cands])
        rules[f"{key}:const"] = const

        step = float(np.mean([v(l[-1]) - v(l[-2]) for l in evidence])) if evidence else 0.0

        def delta(prefix, cands, v=v, scale=scale, step=step):
            pred = v(prefix[-1]) + step
            return np.array([_closeness(pred - v(c), 2 * scale) for c in cands])
        rules[f"{key}:delta"] = delta

        def summed(prefix, cands, v=v, scale=scale):
            if len(prefix) < 2:
                return np.zeros(len(cands))
            pred = v(prefix[0]) + v(prefix[1])
            return np.array([_closeness(pred - v(c), 2 * scale) for c in cands])
        rules[f"{key}:sum"] = summed
    return rules


def simpattern_rules(pan, evidence):
    """The *relationships* inside the last line should mirror the relationships
    inside the complete lines -- a generalisation of the 2017 agent's
    dark-pixel-ratio / intersection-pixel-ratio idea."""
    if not evidence:
        return {}
    depth = len(evidence[0]) - 1
    refs = [float(np.mean([io.sim(pan.raw[l[j]], pan.raw[l[-1]]) for l in evidence]))
            for j in range(depth)]

    def fit(prefix, cands, refs=refs):
        out = []
        for c in cands:
            err = np.mean([abs(io.sim(pan.raw[prefix[j]], pan.raw[c]) - refs[j])
                           for j in range(len(prefix))])
            out.append(max(0.0, 1.0 - err / 0.5))
        return np.array(out)
    return {"simpattern": fit}


def latin_rules(pan, evidence):
    """Latin-square rule: every line holds the same set of attribute values in a
    different order. The answer is whichever value the last line is missing."""
    if not evidence or len(evidence[0]) != 3:
        return {}
    rules = {}
    for kind in io.DESCRIPTORS:
        def fit(prefix, cands, kind=kind):
            if len(prefix) != 2:
                return np.zeros(len(cands))
            d = lambda n: pan.desc(kind, n)
            preds = []
            for ref in evidence:
                best, leftover = -1.0, 0
                for i, j in itertools.permutations(range(3), 2):
                    s = io.tanimoto(d(prefix[0]), d(ref[i])) + io.tanimoto(d(prefix[1]), d(ref[j]))
                    if s > best:
                        best, leftover = s, ({0, 1, 2} - {i, j}).pop()
                preds.append(d(ref[leftover]))
            return np.array([float(np.mean([io.tanimoto(p, d(c)) for p in preds]))
                             for c in cands])
        rules[f"latin:{kind}"] = fit
    return rules


# --------------------------------------------------------------- transforms

def _rot_free(k):
    from scipy import ndimage as _nd
    return lambda a: _nd.rotate(a.astype(np.uint8), k, reshape=False, order=0).astype(bool)


ATTR_TRANSFORMS = dict(io.TRANSFORMS)
ATTR_TRANSFORMS.update({"rot45": _rot_free(45), "rot135": _rot_free(135)})


# --------------------------------------------------------------- families

def _family_makers():
    makers = {}
    for d in DIRECTIONS_3X3:
        makers[f"trans_{d}"] = lambda pan, ev: transform_rules(pan, ev)
        makers[f"logic_{d}"] = logic_rules
        makers[f"num_{d}"] = numeric_rules
        makers[f"simpat_{d}"] = simpattern_rules
        makers[f"attr_{d}"] = lambda pan, ev: {
            f"{k}|{n}": f for k in io.DESCRIPTORS
            for n, f in transform_rules(pan, ev, kind=k).items()}
    makers["latin_row"] = latin_rules
    makers["latin_col"] = latin_rules
    return makers


FAMILY_MAKERS = _family_makers()
FAMILIES = list(FAMILY_MAKERS)
# Each family contributes two already-confidence-scaled votes; the learned
# layer only has to decide how much to trust each family.
FEATURE_NAMES = ([f"{f}__best" for f in FAMILIES] +
                 [f"{f}__soft" for f in FAMILIES] +
                 ["dup_max", "dup_rank", "self_sym", "ink_z"])


def _family_direction(fam):
    return fam.rsplit("_", 1)[1]


VALIDATION_TAU = 0.5      # how sharply a fit advantage translates into belief
FAMILY_GAMMA = 3.0        # how strongly the best-validated family dominates


def _validation_score(fits_true_first, tau=VALIDATION_TAU):
    """Probability the rule assigns to the panel we hid, competing against the
    real answer options.

    Fits are z-scored first so the score is comparable across rule types
    (pixel overlap and 'number of holes' live on very different scales). A rule
    too coarse to separate the options scores ~1/k however well it 'holds',
    which is exactly the behaviour we want from a selector."""
    f = np.asarray(fits_true_first, dtype=float)
    sd = f.std()
    z = (f - f.mean()) / sd if sd > 1e-9 else np.zeros_like(f)
    e = np.exp((z - z.max()) / tau)
    return float(e[0] / e.sum())


class RuleSet:
    """The rules of one family: what each predicts for the answer options, and
    how each performed on the lines we hid from it."""

    __slots__ = ("names", "held_fits", "fits", "_cache")

    def __init__(self, names, held_fits, fits):
        self.names = names
        self.held_fits = held_fits            # [n_rules][n_held](1 + n_choices)
        self.fits = fits                      # (n_rules, n_choices)
        self._cache = {}

    def vconf(self, tau):
        """Mean held-out likelihood per rule."""
        if tau not in self._cache:
            self._cache[tau] = np.array([
                float(np.mean([_validation_score(f, tau) for f in hf]))
                for hf in self.held_fits])
        return self._cache[tau]

    @property
    def agree(self):
        """How well each rule literally held on the lines we can see."""
        return np.array([float(np.mean([f[0] for f in hf])) for hf in self.held_fits])


def score_rules(problem):
    """Returns {family: RuleSet}."""
    pan = Panels(problem)
    choices = problem.choices
    out = {}

    for dname, (completes, prefix) in directions(problem).items():
        for fam, maker in FAMILY_MAKERS.items():
            if _family_direction(fam) != dname:
                continue

            # --- validation: hide each complete line and try to recover it ---
            vfits = {}
            for held in completes:
                evidence = [l for l in completes if l is not held] or completes
                rules = maker(pan, evidence)
                cands = [held[-1]] + choices
                for name, fn in rules.items():
                    vfits.setdefault(name, []).append(fn(held[:-1], cands))

            # --- prediction: all evidence, real target line ---
            rules = maker(pan, completes)
            names, fits, held_fits = [], [], []
            for name, fn in rules.items():
                if name not in vfits:
                    continue
                names.append(name)
                fits.append(fn(prefix, choices))
                held_fits.append(vfits[name])
            if names:
                out[fam] = RuleSet(names, held_fits, np.stack(fits))
    return pan, out


# --------------------------------------------------------------- features

def _znorm(v):
    v = np.asarray(v, dtype=float)
    s = v.std()
    return (v - v.mean()) / s if s > 1e-9 else np.zeros_like(v)


def _family_summary(rs: "RuleSet", tau=VALIDATION_TAU, temperature=0.05):
    """Reduce a family of rules to (best fit, blended fit, weights)."""
    vconf, agree = rs.vconf(tau), rs.agree
    # vconf is already a likelihood; agree only breaks near-ties in its favour
    w = vconf * (0.7 + 0.3 * agree)
    top = int(np.argmax(w))
    zbest = _znorm(rs.fits[top])
    sm = np.exp((w - w.max()) / temperature)
    sm = sm / sm.sum()
    zsoft = _znorm(sm @ rs.fits)
    return zbest, zsoft, vconf[top], agree[top], w[top]


def family_terms(problem, scored, tau=VALIDATION_TAU, gamma=FAMILY_GAMMA):
    """Per-family votes, already scaled by how much the validation pass trusts
    the family. Shared by the no-training solver and the learned ranker."""
    n = problem.n_choices
    best, soft = [], []
    for fam in FAMILIES:
        if fam not in scored:
            best.append(np.zeros(n)); soft.append(np.zeros(n))
            continue
        zbest, zsoft, v, a, w = _family_summary(scored[fam], tau)
        g = w ** gamma
        best.append(g * zbest); soft.append(g * zsoft)
    return best, soft


def feature_matrix(problem, scored=None, tau=VALIDATION_TAU, gamma=FAMILY_GAMMA):
    pan, scored = scored if scored else score_rules(problem)
    n = problem.n_choices
    best, soft = family_terms(problem, scored, tau, gamma)

    givens = [pan.raw[g] for g in problem.givens]
    dup = np.array([max(io.sim(pan.raw[c], g) for g in givens) for c in problem.choices])
    dup_rank = np.argsort(np.argsort(-dup)) / max(n - 1, 1)
    self_sym = np.array([0.5 * (io.symmetry_h(pan.raw[c]) + io.symmetry_v(pan.raw[c]))
                         for c in problem.choices])
    ink_z = _znorm([io.ink(pan.raw[c]) for c in problem.choices])

    return np.column_stack(best + soft +
                           [_znorm(dup), dup_rank, _znorm(self_sym), ink_z])


def unsupervised_score(problem, scored=None, tau=VALIDATION_TAU, gamma=FAMILY_GAMMA):
    """No-training answer: every family votes, weighted by how well its best
    rule survived validation. This is the pure knowledge-based-AI agent."""
    _, scored = scored if scored else score_rules(problem)
    best, soft = family_terms(problem, scored, tau, gamma)
    return 0.6 * np.sum(best, axis=0) + 0.4 * np.sum(soft, axis=0)
