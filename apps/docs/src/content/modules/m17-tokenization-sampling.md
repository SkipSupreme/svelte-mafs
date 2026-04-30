---
course: ml-math
arc: arc-4-capstone
order: 17
title: Tokenization, Training & Sampling
summary: How text becomes integer ids (BPE). How the M16 forward pass becomes a working model (the training loop with AdamW, warmup, cosine decay, gradient clipping). How a trained model becomes text again (autoregressive sampling with temperature, top-k, top-p — and the KV cache that makes inference O(T)).
status: shipped
estimatedMinutes: 84
prereqs: [m13-training-dynamics, m16-transformer-block]
conceptsCovered:
  [
    token-id,
    vocab-artifact,
    char-tokenizer,
    word-tokenizer,
    subword-motivation,
    bpe-train,
    bpe-encode,
    byte-bpe,
    tok-failure-modes,
    corpus-stream,
    data-loader,
    train-val-split,
    loss-everywhere,
    train-loop,
    grad-clip,
    warmup-cosine,
    loss-trajectory,
    overfit-signal,
    autoregressive-loop,
    greedy-fail,
    temperature,
    top-k,
    top-p,
    sampler-pipeline,
    beam-search-not,
    kv-cache-callback,
    seed-determinism,
    prompt-control,
  ]
endgameConnection: 'A tokenizer makes text countable. A loop makes a model learnable. A sampler makes a model talk. Next module: do it for real.'
---
