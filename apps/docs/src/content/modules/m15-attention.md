---
course: ml-math
arc: arc-3-transformers
order: 15
title: Attention
summary: Build attention from a soft dictionary lookup. Scaled dot-product. Q, K, V as projections of the same X. Causal masking. Permutation-equivariance and three flavors of positional encoding. Multi-head as parallel subspaces. The T² cost and the KV-cache that tames it.
status: shipped
estimatedMinutes: 77
prereqs: [m13-training-dynamics, m14-sequence-models]
conceptsCovered:
  [
    weighted-average-as-aggregator,
    soft-dictionary-lookup,
    query-key-value-roles,
    dot-product-as-similarity,
    unscaled-attention-score,
    softmax-saturation-problem,
    1-over-sqrt-dk-scale,
    attention-weights-row,
    single-head-attention-output,
    learned-projections-WqWkWv,
    self-attention,
    attention-formula-matrix-form,
    shape-calculus-attention,
    permutation-equivariance,
    positional-encoding-need,
    sinusoidal-pe,
    learned-absolute-pe,
    rope-rotary-pe,
    causal-mask,
    multi-head-split,
    multi-head-concat-and-Wo,
    attention-quadratic-complexity,
    kv-cache,
    attention-not-explanation,
  ]
endgameConnection: 'Attention is one operation: every position broadcasts a query, every position offers a key, the softmax-weighted match decides whose values you pull back. Residuals, layer norm, and the MLP that follows are plumbing around that core idea — and you are now one layer of glue away from a real transformer.'
---
