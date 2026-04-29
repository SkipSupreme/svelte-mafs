---
course: ml-math
arc: arc-3-transformers
order: 16
title: The Transformer Block
summary: Compose attention with a position-wise FFN, wrap each in residual + layer norm, stack N times, top with a final LN and a tied unembedding. Pre-LN vs post-LN. The residual stream as the noun the model operates on. Where the parameters and FLOPs actually go.
status: shipped
estimatedMinutes: 75
prereqs: [m13-training-dynamics, m14-sequence-models, m15-attention]
conceptsCovered:
  [
    pre-ln-block,
    post-ln-block,
    residual-stream,
    position-wise-ffn,
    four-x-expansion,
    gelu-activation,
    sub-layer-dropout,
    pre-vs-post-ln-gradient,
    block-parameter-budget,
    block-flops,
    stacked-blocks,
    depth-specialization,
    final-layer-norm,
    unembedding,
    weight-tying,
    full-forward-pass,
    model-parameter-budget,
    bigram-as-zero-layer,
  ]
endgameConnection: 'Tokens at the bottom. Attention and MLPs in the middle. Unembedding at the top. Bigram NLL on the outside. Stack N, train with Adam, warm up the LR. That is GPT — there is nothing else.'
---
