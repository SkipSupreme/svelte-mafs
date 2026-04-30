#!/usr/bin/env python3
"""nanoGPT reference oracle for the M18 capstone — Slice 2.

Trains the locked architecture (n_layer=4, n_head=4, d_model=64, d_ff=256,
T=64, vocab=65, tied unembedding, no biases, dropout=0.0, GELU(approximate='tanh'))
on tiny-shakespeare-char for 2,000 iters on CPU.

Emits docs/research/m18-nanogpt-reference.csv with columns
    iter, train_nll, val_nll, lr

This CSV is the gate for Slice 3: the WGSL training loop must match this
trajectory within plus-or-minus 0.1 nats at every 100-iter checkpoint.

Seeds: a single int, threaded through torch + Python random + numpy. The
seed is recorded in docs/research/m18-nanogpt-reference.README.txt.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- locked config ------------------------------------------------------------

VOCAB = 65
T_CTX = 64
D_MODEL = 64
N_HEAD = 4
N_LAYER = 4
D_FF = 256
DROPOUT = 0.0

BATCH = 32
LR_MAX = 3e-4
LR_MIN = 3e-5
WARMUP_ITERS = 100
TOTAL_ITERS = 2000
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
EVAL_EVERY = 100
EVAL_BATCHES = 20

SEED = 1337


# ---- model --------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer(
            'mask',
            torch.tril(torch.ones(T_CTX, T_CTX)).view(1, 1, T_CTX, T_CTX),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate='tanh'))


class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(VOCAB, D_MODEL)
        self.wpe = nn.Embedding(T_CTX, D_MODEL)
        self.blocks = nn.ModuleList([Block(D_MODEL, N_HEAD, D_FF) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, VOCAB, bias=False)
        self.lm_head.weight = self.wte.weight  # tied

        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)


# ---- data ---------------------------------------------------------------------

DATA_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'


def load_corpus(path: Path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f'  downloading tiny-shakespeare -> {path}')
        urllib.request.urlretrieve(DATA_URL, path)
    text = path.read_text()
    chars = sorted(list(set(text)))
    assert len(chars) == VOCAB, f'expected vocab {VOCAB}, got {len(chars)}'
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    n = int(0.9 * len(data))
    return data[:n], data[n:], stoi


def get_batch(data: np.ndarray, B: int, T: int, rng: np.random.Generator):
    ix = rng.integers(0, len(data) - T - 1, size=B)
    x = np.stack([data[i : i + T] for i in ix])
    y = np.stack([data[i + 1 : i + 1 + T] for i in ix])
    return torch.from_numpy(x).long(), torch.from_numpy(y).long()


# ---- LR schedule --------------------------------------------------------------

def lr_at(step: int) -> float:
    if step < WARMUP_ITERS:
        return LR_MAX * (step + 1) / WARMUP_ITERS
    p = (step - WARMUP_ITERS) / max(1, TOTAL_ITERS - WARMUP_ITERS)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1.0 + math.cos(math.pi * p))


# ---- driver -------------------------------------------------------------------

@torch.no_grad()
def measure_val_nll(model: nn.Module, data: np.ndarray, rng: np.random.Generator) -> float:
    model.eval()
    losses = []
    for _ in range(EVAL_BATCHES):
        x, y = get_batch(data, BATCH, T_CTX, rng)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[2] / 'docs/research/m18-nanogpt-reference.csv')
    ap.add_argument('--data', type=Path, default=Path(__file__).resolve().parents[1] / 'm18/tinyshakespeare.txt')
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--iters', type=int, default=TOTAL_ITERS)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_data, val_data, _ = load_corpus(args.data)

    model = TinyGPT()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  model params: {n_params:,}')

    opt = torch.optim.AdamW(model.parameters(), lr=LR_MAX, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY)

    train_rng = np.random.default_rng(args.seed)
    val_rng = np.random.default_rng(args.seed + 1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['iter', 'train_nll', 'val_nll', 'lr'])
        f.flush()

        run_avg = 0.0
        run_count = 0
        t0 = time.time()
        for step in range(args.iters):
            for g in opt.param_groups:
                g['lr'] = lr_at(step)

            x, y = get_batch(train_data, BATCH, T_CTX, train_rng)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            run_avg = 0.9 * run_avg + 0.1 * loss.item() if run_count else loss.item()
            run_count += 1

            if step % EVAL_EVERY == 0 or step == args.iters - 1:
                vnll = measure_val_nll(model, val_data, val_rng)
                w.writerow([step, f'{run_avg:.6f}', f'{vnll:.6f}', f'{lr_at(step):.6e}'])
                f.flush()
                dt = time.time() - t0
                print(f'  iter {step:>5} | train {run_avg:.4f} | val {vnll:.4f} | lr {lr_at(step):.2e} | {dt:6.1f}s')

    print(f'wrote {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
