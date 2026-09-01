"""A relation network for Raven's matrices, in the WReN style.

Architecture, and why it is shaped this way:

    each panel  -> small CNN -> 128-d embedding, plus a one-hot tag for which
                   cell of the grid it sits in
    every pair  -> a shared MLP g(e_i, e_j) -- the relation module
    sum of pairs-> a second MLP f(.) -> one score for this candidate answer
    softmax over candidates, cross-entropy against the true one

The pairwise sum is the whole idea: a rule in a Raven's matrix is a statement
about how two cells relate, so the model is built to compute a representation
of every pair and let the head decide which relations matter. A plain CNN over
the concatenated grid has to discover that structure for itself and, with this
little data, does not.

It is trained only on synthetic matrices from render.py. The 96 real problems
are never seen during training.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CELLS = 9
POS_3X3 = list(range(9))          # A..H then the answer slot
POS_2X2 = [0, 1, 3, 4]            # top-left quadrant: A B / C ?


class PanelEncoder(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        ch = [1, 32, 32, 64, 64]
        layers = []
        for i in range(4):
            layers += [nn.Conv2d(ch[i], ch[i + 1], 3, stride=2, padding=1),
                       nn.BatchNorm2d(ch[i + 1]), nn.ReLU(inplace=True)]
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(64 * 4 * 4, dim)

    def forward(self, x):                       # (N, 1, 64, 64)
        h = self.conv(x).flatten(1)
        return F.relu(self.fc(h))


class RelationNet(nn.Module):
    def __init__(self, dim=128, hidden=256, dropout=0.3):
        super().__init__()
        self.enc = PanelEncoder(dim)
        d = dim + N_CELLS
        self.g = nn.Sequential(nn.Linear(2 * d, hidden), nn.ReLU(inplace=True),
                               nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        self.f = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
                               nn.Dropout(dropout), nn.Linear(hidden, 1))
        iu = torch.triu_indices(N_CELLS, N_CELLS, offset=1)
        self.register_buffer("pair_i", iu[0])
        self.register_buffer("pair_j", iu[1])

    def forward(self, panels, slots, mask, n_ctx):
        """panels (B, P, 1, 64, 64) -- context panels then option panels
           slots  (B, N_CELLS) long -- which grid cell each filled slot uses
           mask   (B, N_CELLS) float -- 1 where the slot is used
           n_ctx  int -- how many of the P panels are context"""
        B, P = panels.shape[:2]
        emb = self.enc(panels.reshape(B * P, 1, 64, 64)).reshape(B, P, -1)
        ctx, opts = emb[:, :n_ctx], emb[:, n_ctx:]
        C = opts.shape[1]

        # build, for each candidate, the full set of N_CELLS slot embeddings
        dim = emb.shape[-1]
        grid = torch.zeros(B, C, N_CELLS, dim, device=emb.device, dtype=emb.dtype)
        # slots[:, :n_ctx] are the context cells; slots[n_ctx] is the answer cell
        idx_ctx = slots[:, :n_ctx]                             # (B, n_ctx)
        grid.scatter_(2, idx_ctx[:, None, :, None].expand(B, C, n_ctx, dim),
                      ctx[:, None].expand(B, C, n_ctx, dim))
        ans_slot = slots[:, n_ctx]                             # (B,)
        grid.scatter_(2, ans_slot[:, None, None, None].expand(B, C, 1, dim),
                      opts.unsqueeze(2))

        tag = torch.eye(N_CELLS, device=emb.device, dtype=emb.dtype)
        grid = torch.cat([grid, tag.expand(B, C, N_CELLS, N_CELLS)], dim=-1)

        gi, gj = grid[:, :, self.pair_i], grid[:, :, self.pair_j]
        pair_ok = (mask[:, self.pair_i] * mask[:, self.pair_j])[:, None, :, None]
        rel = self.g(torch.cat([gi, gj], dim=-1)) * pair_ok
        pooled = rel.sum(2) / pair_ok.sum(2).clamp(min=1)
        return self.f(pooled).squeeze(-1)                      # (B, C)


# --------------------------------------------------------------- data

def pack(context, options, three):
    """Turn one problem into fixed-size arrays the model can batch."""
    cells = POS_3X3 if three else POS_2X2
    n_ctx = len(cells) - 1
    slots = np.zeros(N_CELLS, dtype=np.int64)
    slots[:n_ctx] = cells[:n_ctx]
    slots[n_ctx] = cells[-1]
    mask = np.zeros(N_CELLS, dtype=np.float32)
    mask[cells] = 1.0
    panels = np.stack([p.astype(np.float32) for p in list(context) + list(options)])
    panels = 1.0 - panels / 255.0                              # ink = 1
    return panels, slots, mask, n_ctx
