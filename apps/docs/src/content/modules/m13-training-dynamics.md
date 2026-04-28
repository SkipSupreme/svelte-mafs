---
course: ml-math
arc: arc-2-ml-foundations
order: 13
title: Training Dynamics & Modern Tricks
summary: Over/underfitting. Train/val/test splits. L2. Dropout. Weight init. BatchNorm. LayerNorm. Residual connections. LR warmup.
status: drafting
estimatedMinutes: 180
prereqs: [m10-optimization, m11-neural-networks, m12-backpropagation]
conceptsCovered: [overfitting, regularization, dropout, weight-init, batchnorm, layernorm, residual, lr-warmup]
endgameConnection: 'The transformer block you''re about to build is `x = x + F(LN(x))` repeated N times. Pre-LN keeps activations bounded; the `+ x` is the only reason gradients survive 12 layers; the `0.02/√(2N)` init keeps the residual stream from blowing up; and linear warmup is the only reason AdamW doesn''t take a wild first step. Every line is something from this module.'
---
