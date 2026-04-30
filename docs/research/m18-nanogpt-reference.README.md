# M18 nanoGPT reference oracle

`m18-nanogpt-reference.csv` is a 2,000-iter loss trajectory of the locked M18
architecture trained on tiny-shakespeare-char with PyTorch on CPU. This is the
gate Slice 3 must match: the WGSL training loop is "right" iff it stays within
±0.1 nats of these `val_nll` checkpoints at every 100-iter mark.

- **Architecture (locked):** `n_layer=4, n_head=4, d_model=64, d_ff=256, T=64,
  vocab=65, tied unembedding, no Linear biases, dropout=0.0,
  GELU(approximate='tanh')`
- **Optimizer:** AdamW, betas=(0.9, 0.95), weight_decay=0.1, grad-clip=1.0
- **LR schedule:** linear warmup 100 iters → cosine decay from 3e-4 to 3e-5
- **Batch:** 32 sequences × T=64
- **Seed:** `1337` (torch + numpy; train and val use independent PRNG streams)
- **Eval:** every 100 iters, mean over 20 batches of held-out val data
- **Source script:** `scripts/m18/nanogpt_reference.py`
- **Corpus:** tiny-shakespeare-char (1,115,394 chars, 90/10 train/val split,
  vocab 65) — `scripts/m18/tinyshakespeare.txt`

To regenerate (CPU, ~minutes):

```sh
python3 scripts/m18/nanogpt_reference.py
```

The CSV columns are `iter, train_nll, val_nll, lr`. `train_nll` is an EWMA
(decay 0.9) of the per-iter cross-entropy loss; `val_nll` is a fresh eval each
checkpoint. They diverge by ~0.05 nat throughout; that's normal.

CPU reproducibility: PyTorch float32 is bit-identical across runs of this
script with the same seed on the same machine. Cross-platform reproducibility
is not a goal of this oracle — Slice 3's gate is the trajectory shape and the
±0.1 nat envelope, not a bit-equality match.
