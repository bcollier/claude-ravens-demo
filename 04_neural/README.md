# Agent D: a neural network (epilogue, still no LLM)

Added after class. The in-class "no LLM" agent used classical CV and a linear
ranker; this is what a neural network does on the same problems.

## The 96-problem problem

You cannot train a network on 96 examples. The way round it — and what the
research literature on this task does — is to generate an unlimited supply of
synthetic matrices, train on those, and treat the 96 real problems as a held-out
test the network has never seen.

- **`render.py`** is the generator. It builds matrices from the same rule
  vocabulary the real sets use — constant, progression, Latin square, arithmetic
  on object counts, and pixel XOR / OR / AND — over twelve shapes chosen to
  cover what the real sets contain, and renders them as black line art the way
  the originals look. Panels are augmented with small jitter and varying stroke
  weight so the network cannot lock onto the renderer's exact geometry.
- **`wren.py`** is the model, a relation network in the style of Santoro et
  al.'s WReN: a small CNN embeds each panel, a one-hot tag says which grid cell
  it occupies, a shared MLP runs over *every pair* of panels, the pair
  representations are summed, and a head scores the candidate. The pairwise sum
  is the whole idea — a rule in a Raven's matrix is a statement about how two
  cells relate, so the architecture computes exactly that.
- **`solver.py`** trains it and evaluates on the real 96.

## Running it

```bash
python 04_neural/solver.py --steps 8000              # train, then test on the real 96
python 04_neural/solver.py --skip-train              # reuse the checkpoint
python 04_neural/solver.py --steps 5000 --tag myrun  # an ablation under its own name
```

Trains on Apple MPS or CPU. No pretrained weights, no downloads.

## The finding worth keeping

The first version of `render.py` gave every attribute its own rule at once,
producing matrices that are visually chaotic in a way real Raven's problems
never are. Fixing the *generator* — holding most attributes fixed and varying
one or two under a rule, as the real sets do — moved held-out synthetic accuracy
far more than any change to the architecture would have. The training
distribution was the bottleneck, not the model.

See [`../EPILOGUE.md`](../EPILOGUE.md) for the numbers and the comparison
against the symbolic agent.
