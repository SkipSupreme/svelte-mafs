# Module 16 — The Transformer Block

## 1. Concept dependency graph

1. `pre-ln-block` — A transformer block that applies LayerNorm to its input *before* each sub-layer and adds the sub-layer's output to the unnormalized residual stream: `x → x + MHA(LN(x)) → x + FFN(LN(x))`. Prereqs: `m13-training-dynamics` (residual + layer norm), `m15-attention` (MHA).
1. `post-ln-block` — The original 2017 variant: `x → LN(x + MHA(x)) → LN(x + FFN(...))`. The norm sits *after* the residual add, on the trunk. Prereqs: `pre-ln-block`.
1. `residual-stream` — The per-token vector that flows top-to-bottom through the stack; every sub-layer reads it (via LN), computes a delta, and writes that delta back additively. Prereqs: `pre-ln-block`, `m13-training-dynamics`.
1. `position-wise-ffn` — A 2-layer MLP applied identically and *independently* to every token's residual-stream vector, with no cross-token mixing. Prereqs: `m11-neural-networks`, `residual-stream`.
1. `four-x-expansion` — The convention `d_ff = 4 · d_model` for the FFN's hidden width. Prereqs: `position-wise-ffn`.
1. `gelu-activation` — The smooth Gaussian-CDF-gated nonlinearity `x · Φ(x)` used in modern FFNs in place of ReLU. Prereqs: `m11-neural-networks`, `position-wise-ffn`.
1. `sub-layer-dropout` — Dropout placed on each sub-layer's *output* (before it is added to the residual stream), and on the post-embedding sum. Prereqs: `pre-ln-block`, `m13-training-dynamics`.
1. `pre-vs-post-ln-gradient` — At initialization, post-LN gradient norm in the final layers grows with depth `L` (requiring warmup); pre-LN gradient norm is `O(1/√L)`-bounded and trains stably without warmup. Prereqs: `pre-ln-block`, `post-ln-block`, `m12-backpropagation`.
1. `block-parameter-budget` — Per-block parameter count: `~4 d²` from {Q,K,V,O} attention projections + `~8 d²` from the two FFN matrices ≈ `12 d²`, FFN-dominated as `d` grows. Prereqs: `pre-ln-block`, `position-wise-ffn`.
1. `block-flops` — Per-token forward FLOPs ≈ `2 · params_per_block ≈ 24 d²`; training adds backward ≈ `4 · params` for `~6N` per token total. Prereqs: `block-parameter-budget`.
1. `stacked-blocks` — `N` identical blocks composed via residual addition; depth determines how many sequential read/write rounds the residual stream undergoes. Prereqs: `residual-stream`.
1. `depth-specialization` — Empirical finding that early blocks tend to handle surface/positional features, middle blocks relational/semantic, late blocks task/output-shape features. Prereqs: `stacked-blocks`.
1. `final-layer-norm` — The single LayerNorm applied at the top of the stack (after the last block) before the LM head — required because pre-LN never normalizes the trunk itself. Prereqs: `stacked-blocks`, `pre-ln-block`.
1. `unembedding` — A linear projection `W_U: ℝ^d → ℝ^V` from the residual-stream vector at each position to next-token logits. Prereqs: `m14-sequence-modeling` (vocab, embedding lookup).
1. `weight-tying` — Setting `W_U = W_E^T`, sharing parameters between input embedding and output projection. Prereqs: `unembedding`, `m14-sequence-modeling`.
1. `full-forward-pass` — `tokens → embed + posenc → N × pre-LN block → final-LN → unembed → softmax → next-token distribution`. Prereqs: every concept above plus `m15-attention`.
1. `model-parameter-budget` — Total ≈ `2 · V · d` (tied embeddings) + `12 · N · d²`. For tied GPT-2-small (V=50257, d=768, N=12): ~124M params. Prereqs: `block-parameter-budget`, `weight-tying`.
1. `bigram-as-zero-layer` — Setting `N=0` reduces the model to `softmax(W_U · embed(t))`, which is exactly the bigram model of m14 — the floor of the architecture. Prereqs: `full-forward-pass`, `m14-sequence-modeling`.

## 2. Canonical worked examples

### Example A — Trace one residual stream through a 2-block toy

**Problem.** Let `d_model = 4`, `N = 2`, sequence `["a","b"]`. Given an embedding `x_0 = (1, 0, -1, 0)` for token `b`, an attention sub-layer that contributes `Δ_attn1 = (0, 0.5, 0, 0)`, an FFN that contributes `Δ_ffn1 = (0, 0, 0.25, 0.25)`, and analogous `Δ_attn2 = (0.1, 0, 0, -0.1)`, `Δ_ffn2 = (0, 0.2, 0, 0)`, write the residual stream after each sub-layer in pre-LN form.

**Solution.**

- `x_1a = x_0 + Δ_attn1 = (1, 0.5, -1, 0)`
- `x_1b = x_1a + Δ_ffn1 = (1, 0.5, -0.75, 0.25)` (output of block 1)
- `x_2a = x_1b + Δ_attn2 = (1.1, 0.5, -0.75, 0.15)`
- `x_2b = x_2a + Δ_ffn2 = (1.1, 0.7, -0.75, 0.15)` (output of block 2)
- After final LN with γ=1, β=0, ε=0: mean = 0.05, var ≈ 0.572, σ ≈ 0.756; `x_final ≈ (1.388, 0.860, -1.058, 0.132)`.

**Pedagogical point.** A block does not *transform* the stream; it adds a delta. Composition is summation. This is the linchpin of the Anthropic "residual stream as the object" framing.

**Common mistake.** Students apply the LN inside the addition (`LN(x + Δ)`) — that's post-LN. In pre-LN, the LN is upstream of the sub-layer only; the residual trunk is never normalized between blocks.

### Example B — Parameter count of GPT-2 small

**Problem.** Given `V = 50257`, `d = 768`, `n_heads = 12`, `n_layers = 12`, `context = 1024`, `d_ff = 4d`, with weight tying. Compute total parameters.

**Solution.**

- Token embedding: `V · d = 50257 · 768 = 38,597,376`. (Tied — counted once.)
- Learned positional: `1024 · 768 = 786,432`.
- Per block:
  - Attention `{W_Q, W_K, W_V, W_O}`: `4 · d² = 4 · 768² = 2,359,296`
  - FFN `{W_1, W_2}`: `2 · d · 4d = 8 · 768² = 4,718,592`
  - 2 LayerNorms: `2 · 2 · d = 3,072` (negligible)
  - Block total ≈ `7,081,000` (≈ `12 d²` if you ignore biases & LN)
- 12 blocks: `≈ 84,973,000`
- Final LN: `2 · 768 = 1,536` (negligible)
- Total ≈ `38,597,376 + 786,432 + 84,973,000 + ~1,500 ≈ 124.4 M`.

**Pedagogical point.** ~70% of the *non-embedding* parameters live in the FFN, not attention. The attention is the thing the textbooks tattoo on the cover; the FFN is where the weight is.

**Common mistake.** Forgetting weight tying and double-counting `V·d`, or counting the LM head separately. With tying, the LM head reuses `W_E^T`.

### Example C — Pre-LN vs post-LN gradient at depth (Xiong 2020 sketch)

**Problem.** Show heuristically why post-LN gradients near the *output* layer scale like `O(√L)` (so warmup is needed) while pre-LN gradients are `O(1)` in `L`.

**Solution sketch.**

- Post-LN: `x_{ℓ+1} = LN(x_ℓ + F_ℓ(x_ℓ))`. At init, `‖x_ℓ + F_ℓ(x_ℓ)‖²` ≈ `‖x_ℓ‖² + ‖F_ℓ‖²`, and LN renormalizes back to unit scale. The gradient of the loss w.r.t. parameters of layer `ℓ` carries a factor that, summed over `L` layers, scales as `√L` near the top — large gradients on the last block at step 0.
- Pre-LN: `x_{ℓ+1} = x_ℓ + F_ℓ(LN(x_ℓ))`. The Jacobian `∂x_{ℓ+1}/∂x_ℓ` is `I + (small)`. Walking back through `L` such Jacobians keeps gradient norm `O(1)`. In the formal analysis (Xiong et al. 2020, Theorem 1), the expected gradient of the parameters at the final block is `Θ(d/√L · ln L)` for pre-LN vs `Θ(d · ln L)` for post-LN.
- Practical consequence: pre-LN trains stably with constant LR; post-LN explodes without a warmup of ~4k–10k steps.

**Pedagogical point.** The placement of LN is *not* a stylistic choice; it is the difference between needing warmup and not.

**Common mistake.** Concluding "pre-LN is strictly better." It isn't — final loss is sometimes slightly worse than well-tuned post-LN, and recent work (Peri-LN, DeepNorm) re-opens the question. But for *teaching* and for *deep* models, pre-LN is the right default.

### Example D — FLOPs for one forward pass of a single block

**Problem.** With `d = 768`, sequence length `T = 1024`, count forward FLOPs (multiply-adds × 2) for one block.

**Solution.**

- Q, K, V projections: `3 · 2 · T · d² = 6 · 1024 · 768² ≈ 3.62 · 10⁹`
- `QK^T`: `2 · T² · d = 2 · 1024² · 768 ≈ 1.61 · 10⁹`
- Softmax-weighted V: `2 · T² · d ≈ 1.61 · 10⁹`
- Output projection: `2 · T · d² ≈ 1.21 · 10⁹`
- FFN: `2 · 2 · T · d · 4d = 16 · T · d² ≈ 9.66 · 10⁹`
- Block total: `≈ 17.7 GFLOPs`. FFN ≈ 55%, attention projections ≈ 27%, attention matmul ≈ 18%.

**Pedagogical point.** At `T = d`, FFN and attention-projection FLOPs are comparable; the `T²` term only dominates when `T » d`.

**Common mistake.** Conflating `T²` (the famous quadratic) with overall cost. For GPT-2-small at `T=1024`, the `T²` part is < 20% of a block's FLOPs.

### Example E — Stacking N blocks: the bigram floor

**Problem.** Show that with `N = 0`, the architecture is exactly the m14 bigram model.

**Solution.** Forward pass with N=0: `logits = LN(W_E[t] + p_t) · W_U`. With weight tying and ignoring positional encoding (set to 0) and LN (γ=1, β=0 trivially): `logits ≈ W_E[t] · W_E^T`. Softmax over this gives the bigram distribution `P(next | t)`. Trained with NLL, this is exactly the m14 bigram-as-one-layer-NN.

**Pedagogical point.** Every transformer is a bigram model with a stack of additive corrections (`Δ_ℓ`) added to the residual stream. Each block buys you context-dependent edits to the bigram baseline.

**Common mistake.** Believing the transformer "throws away" the bigram channel. It doesn't — the direct path `W_E → W_U` is always present (Anthropic's "direct path" term), and circuits actually exploit it.

### Example F — The 4× expansion ratio

**Problem.** Why `d_ff = 4d` and not `2d` or `8d`?

**Solution.** Set `d_ff = r · d`. Block params from FFN: `2 r d²`. Block params from attention: `4 d²`. Total: `(2r + 4) d²`. Attention ratio: `4 / (2r + 4)`. For `r = 4`, attention is `4/12 = 33%`, FFN is `67%`. Empirically, `r = 4` was used in Vaswani 2017, kept by GPT-1/2/3, BERT, T5; ablations in subsequent papers showed `r ∈ [2, 8]` performs comparably *per parameter* with quality maximizing near 4 in standard recipes. Modern gated variants (SwiGLU in LLaMA) use `r ≈ 8/3` because the gate adds a third matrix.

**Pedagogical point.** The 4× isn't sacred geometry; it's a 2017-vintage convention that makes FFN ≈ 2× attention params, kept because it works.

**Common mistake.** "More FFN width is always better." No — ablations show diminishing returns past `r = 4` for dense FFNs at fixed total params.

### Example G — Weight tying impact on a tiny model

**Problem.** Compare an untied vs tied character-level model with `V = 65`, `d = 64`, `N = 4`.

**Solution.**

- Untied embedding + LM head: `2 · V · d = 8,320` params.
- Tied: `V · d = 4,160`.
- Block params: `12 · d² · N = 12 · 64² · 4 = 196,608`.
- Tied total ≈ `200,768`; untied ≈ `204,928`. Tying saves ~2% here.
- But for GPT-2-small: tying saves `V · d = 38.6 M` out of ~163 M untied → ~24%.

**Pedagogical point.** Tying matters more for small models and large vocabularies. It's also a regularizer: input and output spaces are forced to share geometry, which Press & Wolf 2017 showed lowers perplexity.

**Common mistake.** Thinking tying is "just to save memory." It changes loss landscape and inductive bias.

### Example H — Identifying a pre-LN block from PyTorch code

**Problem.** Given:

```
def block(x):
    a = self.ln1(x)
    a = self.attn(a)
    x = x + self.dropout(a)
    b = self.ln2(x)
    b = self.ffn(b)
    x = x + self.dropout(b)
    return x
```

Is this pre-LN or post-LN? Where is dropout placed?

**Solution.** Pre-LN (LN comes before each sub-layer; residual trunk `x` is never normalized). Dropout is on the sub-layer *output*, before the residual add. There is *no* dropout on the residual trunk itself. This matches the standard recipe (and the original 2017 paper for the dropout placement: `Sublayer(x) → dropout → +x`).

**Pedagogical point.** You can read pre/post-LN off the code by asking: "does `x` (the residual trunk) ever pass *through* an `LN` before being added to?" If no → pre-LN.

**Common mistake.** Confusing "the LN happens first in the line numbers" with "post-LN." Order of *code* doesn't determine the architecture; data flow does.

## 3. Common misconceptions

1. **"The FFN mixes information across tokens."**
   *Why natural:* The FFN is the biggest matrix in the block; surely it's doing the heavy interaction work.
   *Fix:* The FFN has no token axis. It's `MLP(x[t])` for each `t` independently — literally `nn.Linear(d, 4d)` applied along the last dim. Build a widget that lets the user shuffle tokens *only after* the attention sub-layer; the FFN's output permutes identically. Cross-token mixing is exclusively attention's job.
1. **"The transformer is the attention. The FFN is a detail."**
   *Why natural:* "Attention Is All You Need" is the marketing.
   *Fix:* Show the parameter pie. ~67% of non-embedding params and ~55% of FLOPs are FFN. Geva et al. 2020 ("Feed-Forward Layers Are Key-Value Memories") shows the FFN is where factual associations live. Anthropic's circuits work studies attention because it's *easier*, not because it's *most*.
1. **"Layer norm is just normalization. Where you put it doesn't matter."**
   *Why natural:* In CNNs with BatchNorm the placement is largely interchangeable.
   *Fix:* Pre-LN trains without warmup; post-LN at the same depth diverges without it. Run the gradient-norm widget side by side at L=24 and watch post-LN saturate. Xiong et al. 2020 made this rigorous.
1. **"Deeper means more attention."**
   *Why natural:* Each new block adds an attention sub-layer.
   *Fix:* Each new block also adds an FFN that is twice as wide. Scaling up depth scales FFN params 2× faster than attention params. The "more attention" intuition is empirically wrong; modern scaling adds proportional capacity to both.
1. **"Weight tying is just a memory hack."**
   *Why natural:* It saves `V·d` parameters; that sounds purely economical.
   *Fix:* Press & Wolf 2017 showed tied models have *lower* perplexity than untied, controlling for params. It's a regularizer + an inductive bias: the model is forced to use the same geometric direction to "represent" and to "predict" a token.
1. **"The residual stream is just plumbing."**
   *Why natural:* "Skip connection" sounds plumbing-ish.
   *Fix:* Adopt the Anthropic framing: the residual stream is the *object*, and blocks are read/write operations on it. Show that for any two non-adjacent layers, you can decompose their interaction as a sum of paths through the stream. Circuits live here, not in the blocks.
1. **"More heads is always better."**
   *Why natural:* "More" parallel anything in deep learning sounds good.
   *Fix:* Heads share the residual-stream bandwidth `d`. Each head writes into a `d/n_heads`-dim subspace via `W_O`. With fixed `d`, doubling `n_heads` halves `d_head`, eventually starving each head. Empirically the sweet spot is `d_head ∈ [64, 128]`. The bandwidth point comes from Elhage et al. 2021.
1. **"You need a final softmax inside the block."**
   *Why natural:* Attention has one. Output has one. Surely the block has one too.
   *Fix:* The only softmaxes are inside attention's `softmax(QK^T/√d_k)` and at the very top (token logits → distribution). The FFN ends with a linear layer; the block ends with an addition. No softmax on the trunk.
1. **"Pre-LN means LN is somewhere at the front of the model."**
   *Why natural:* The name suggests prefixing.
   *Fix:* Pre-LN means LN is *before each sub-layer's input*, repeated inside every block. Post-LN means LN is *after the residual add*, also inside every block. Both apply twice per block; the difference is which side of the `+` they're on.
1. **"The unembedding is a separate learned matrix unrelated to the embedding."**
   *Why natural:* `embedding` and `unembedding` are different operations.
   *Fix:* In every modern GPT-style model since GPT-2, they are literally the same parameters via tying. The "logit lens" trick (nostalgebraist 2020) only works *because* this is true: any residual-stream vector at any layer can be projected by `W_E^T` to inspect its current "best guess" distribution.
1. **"The 4× expansion is a derived optimum."**
   *Why natural:* Numbers feel principled.
   *Fix:* It's a 2017 convention that subsequent ablations validated as "fine." LLaMA's gated FFN uses 8/3. PaLM uses SwiGLU at different ratios. Treat 4× as a Schelling point, not a theorem.
1. **"Adding dropout everywhere helps."**
   *Why natural:* Dropout is regularization; regularization is good.
   *Fix:* Standard recipe drops only on (a) embedding+positional sum, (b) sub-layer output before the residual add, (c) attention weights. Dropping on the residual *trunk* breaks the gradient highway. Show this with the gradient-flow widget.

## 4. Interactive widget suggestions

### `residualStreamScope`

**Manipulate:** Click any (token-position, layer) cell in a 2D grid for a tiny pre-trained model (e.g., 4-layer, d=64, char-level). Drag a "decomposition" toggle: raw stream / direct path / per-block-Δ contributions / per-head-Δ contributions.
**Live updates:** A horizontal bar of 64 cells colored by signed activation magnitude; an inline "logit lens" decoded top-5 tokens for that residual-stream vector; a stacked-bar showing how much each previous sub-layer contributed to each dimension.
**Concept it makes tangible:** The residual stream is a real, manipulable mathematical object — not a wire. Clicking shows that block 2's Δ wrote `+0.7` into dim 17 and 0 elsewhere, etc.
**Why it beats a slider:** A slider gives a number. This gives the *vector and its provenance* — you can literally watch the model "decide" by rounds of additive edits.
**Prior art:**

- Brendan Bycroft's LLM Visualization (https://bbycroft.net/llm) — 3D walkthrough that exposes per-step vectors but isn't editable.
- Transformer Explainer (https://poloclub.github.io/transformer-explainer/) — runs GPT-2 in browser, shows component values; closer to a guided tour than a manipulable object.
- The logit lens demos (nostalgebraist 2020; Belrose et al.'s tuned-lens, https://github.com/AlignmentResearch/tuned-lens) — apply `W_U` to intermediate streams.

### `lnPlacementGrad`

**Manipulate:** Drag a depth slider (1 to 48) and a toggle (pre-LN ↔ post-LN). Optionally: drag the LN's γ initialization.
**Live updates:** Two superimposed curves — gradient norm at each layer at *step 0* — animated as the user drags depth. A "training simulation" sub-widget runs 200 SGD steps on a tiny task with each architecture; learning curves diverge dramatically without warmup for post-LN.
**Concept it makes tangible:** The pre-LN argument as a felt experience: the user sees the post-LN gradient curve detonate as depth grows.
**Why it beats a slider:** It's the curve *at every depth simultaneously* + the actual training-loss consequence. No static figure can show how the entire profile bends with depth.
**Prior art:** Xiong et al. 2020 figures (paper-only static); APXML's "Normalization Layer Placement" page (interactive walkthroughs but no live training).

### `paramBudgetPie`

**Manipulate:** Drag four handles around a sunburst: `d_model`, `n_heads`, `n_layers`, `vocab`. Optional toggles: weight tying on/off, FFN ratio (2/4/8).
**Live updates:** A live pie/sunburst slicing total params into {tied embedding, attention QKVO, FFN W1+W2, layernorm, positional, LM head}; running totals update; a "GPT-2 small / GPT-2 medium / GPT-3 175B" preset row that snaps the sliders.
**Concept it makes tangible:** "FFN dominates as `d` grows." Drag `d` from 64 to 4096 and watch FFN's slice eat the embedding's slice.
**Why it beats a slider:** Sliders give one number; this gives the geometric shape of the parameter budget. The Bret Victor test: the user is grabbing the *budget*, not adjusting it.
**Prior art:** Karpathy's nanoGPT `scaling_laws.ipynb` (https://github.com/karpathy/nanoGPT/blob/master/scaling_laws.ipynb) — same arithmetic, but in code; not live. Brenndoerfer's interactive GPT-2 piece has a static pie.

### `ablationLab`

**Manipulate:** A pre-trained tiny char-level model is rendered as a stack of blocks. The user clicks individual sub-layer boxes (head, FFN, LN) to ablate them (zero out that sub-layer's contribution to the residual stream).
**Live updates:** Live NLL on a held-out shakespeare passage; per-token loss heatmap; sample generations.
**Concept it makes tangible:** Removing one FFN sub-layer typically *worse* than removing one attention head. Removing the final LN catastrophic. Surfaces FFN dominance and circuit-level redundancy.
**Why it beats a slider:** The action is "click the actual sub-layer you want to kill," not "set ablation strength." Direct manipulation of the computational object.
**Prior art:** TransformerLens (https://github.com/TransformerLensOrg/TransformerLens) and CircuitsVis (https://github.com/TransformerLensOrg/CircuitsVis) — Python-only, not in-browser-interactive; same idea pulled from Anthropic's hook-and-ablate methodology.

### `buildTheBlock`

**Manipulate:** A drag-and-drop palette of {LN, MHA, FFN, residual-add, dropout} blocks and an empty block schematic. The user wires them.
**Live updates:** The graph is parsed into a DAG. The system labels the result: ✓ valid pre-LN block / ✓ valid post-LN block / ✗ broken (missing residual / norm on trunk / etc.). Three optional test-case generations show whether the assembled block, randomly initialized at depth 24, would train without warmup.
**Concept it makes tangible:** There are exactly two canonical wirings, and only the "trunk-untouched" one (pre-LN) is forgiving at depth.
**Why it beats a slider:** Construction, not choice. The student has to *build* the block to learn what makes it correct. This is the assembly module's central act.
**Prior art:** None I could verify on the live web for a transformer block specifically. The closest analogue is the "build a CNN" section of the TensorFlow Playground (https://playground.tensorflow.org), which is layer-level, not sub-layer-level.

### `directPathInspector`

**Manipulate:** Click any token in the input; drag a "block budget" slider from 0 to N.
**Live updates:** Show the model's prediction with only the first `k` blocks active and the rest replaced by identity. At `k=0`, the prediction is *exactly* the bigram from m14 (with weight tying, this is `softmax(W_E[t] · W_E^T)`). At `k=N`, full model.
**Concept it makes tangible:** The blocks are corrections layered onto a bigram. The bigram is the floor; depth buys context-dependent refinements.
**Why it beats a slider:** It connects m14's bigram directly to m16's full stack — a structural, not numerical, link. Distills Anthropic's "direct path" framing.
**Prior art:** Logit-lens visualizations (nostalgebraist's original notebook; https://github.com/AlignmentResearch/tuned-lens) show this layer-by-layer trajectory but don't expose the `k=0` bigram floor cleanly.

## 5. Key formulas

### Pre-LN block (canonical, recommended default)

```latex
\tilde x = x + \mathrm{Drop}\!\big(\mathrm{MHA}(\mathrm{LN}(x))\big)
```

```latex
y = \tilde x + \mathrm{Drop}\!\big(\mathrm{FFN}(\mathrm{LN}(\tilde x))\big)
```

### Post-LN block (original 2017 variant)

```latex
\tilde x = \mathrm{LN}\!\big(x + \mathrm{Drop}(\mathrm{MHA}(x))\big)
```

```latex
y = \mathrm{LN}\!\big(\tilde x + \mathrm{Drop}(\mathrm{FFN}(\tilde x))\big)
```

### Position-wise FFN with GELU

```latex
\mathrm{FFN}(z) = \mathrm{GELU}\!\big(z W_1 + b_1\big)\, W_2 + b_2
\quad\text{with}\quad W_1 \in \mathbb{R}^{d \times 4d},\ W_2 \in \mathbb{R}^{4d \times d}
```

### LayerNorm

```latex
\mathrm{LN}(x)_i = \gamma_i \cdot \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta_i,
\quad \mu = \tfrac{1}{d}\sum_j x_j,\ \sigma^2 = \tfrac{1}{d}\sum_j (x_j - \mu)^2
```

### GELU (exact and tanh-approximation)

```latex
\mathrm{GELU}(x) = x\,\Phi(x) = \tfrac{x}{2}\!\left(1 + \mathrm{erf}\!\left(\tfrac{x}{\sqrt{2}}\right)\right)
```

```latex
\mathrm{GELU}(x) \approx \tfrac{x}{2}\!\left(1 + \tanh\!\left[\sqrt{\tfrac{2}{\pi}}\,(x + 0.044715\,x^{3})\right]\right)
```

### Full forward pass (decoder-only, pre-LN, weight-tied)

```latex
h_0 = E[t_{1:T}] + P[1{:}T]
```

```latex
h_\ell = \mathrm{Block}_\ell(h_{\ell-1}), \quad \ell = 1,\dots,N
```

```latex
z = \mathrm{LN}_{\!\text{final}}(h_N)
```

```latex
\mathrm{logits} = z\, W_U, \qquad p(\cdot \mid t_{\le i}) = \mathrm{softmax}(\mathrm{logits}_i)
```

### Weight tying

```latex
W_U = W_E^{\top}
```

### Per-block parameter count (no biases, ignoring LN)

```latex
P_{\text{block}} = \underbrace{4 d^{2}}_{Q,K,V,O} \;+\; \underbrace{2 \cdot d \cdot 4d}_{\text{FFN}} \;=\; 12\, d^{2}
```

### Full-model parameter count (tied, learned positional)

```latex
P_{\text{model}} \approx V d \;+\; T_{\max}\, d \;+\; N \cdot 12\, d^{2} \;+\; 2 d
```

### Per-token forward FLOPs (excluding attention's T² term)

```latex
\mathrm{FLOPs}_{\text{fwd/token}} \approx 2\, P_{\text{non-emb}} \approx 24\, N\, d^{2}
```

### Training FLOPs per token (Kaplan rule of thumb)

```latex
C_{\text{train/token}} \approx 6\, P_{\text{non-emb}}
```

### Attention's quadratic term per token

```latex
\mathrm{FLOPs}_{\text{attn-mix/token}} \approx 4\, T\, d
```

### NLL training objective (recap from m14, applied here)

```latex
\mathcal{L} = -\frac{1}{T}\sum_{i=1}^{T} \log p(t_{i+1} \mid t_{\le i})
```

## 6. Lesson decomposition

Five lessons. Total ~75 minutes of student time.

### Lesson 16.1 — One block, top to bottom

**Summary:** Build a single transformer block from already-known parts and watch a vector flow through it.
**Steps (12):**

1. *Recap residual + LN* — one paragraph; m13 callback.
1. *Two sub-layers, one rule* — every sub-layer is `x = x + Sub(LN(x))`. Static figure.
1. *MHA as the first sub-layer* — point to m15; sub-layer is `MHA(LN(x))`.
1. *FFN as the second sub-layer* — `Linear → GELU → Linear`, applied per-token.
1. *Why per-token?* — StepCheck: attention already mixed; if FFN mixed too, what would you double-count? (Expected: cross-token information flow.)
1. *The 4× expansion* — drag width with `paramBudgetPie`; watch FFN slice grow.
1. *GELU vs ReLU* — interactive plot; point out smoothness; Hendrycks-Gimpel reference.
1. *StepCheck:* compute the output dim of the first FFN matrix when `d_model=512`. (Expected: `2048`.)
1. *Trace one stream through one block* — Worked Example A, abridged to one block.
1. *StepCheck:* given `x = (1, 0, -1)`, `Δ_attn = (0, 0.5, 0)`, `Δ_ffn = (0, 0, 0.25)`, what is `y`? (Expected: `(1, 0.5, -0.75)`.)
1. *Where does dropout sit?* — three slots labeled.
1. *Endgame callback.*
   **Widgets:** `residualStreamScope` (read-only mode), `paramBudgetPie`.
   **Estimated minutes:** 15.

### Lesson 16.2 — Why pre-LN

**Summary:** Two architectures differ by one wire; one trains without warmup, one doesn't. Find out why.
**Steps (10):**

1. *The two diagrams, side by side.*
1. *Read the difference off the equations.* StepCheck: in pre-LN, does `x` ever pass *through* an LN before the next residual add? (Expected: `0` (no).)
1. *Gradient norm at init — pre vs post* — drive `lnPlacementGrad`.
1. *The Xiong 2020 result, in one sentence* — post-LN gradient at top scales as `√L`; pre-LN as `1/√L`.
1. *Warmup, intuitively.* Why does post-LN need it.
1. *But pre-LN has a price* — final loss can be marginally worse; final LN is mandatory.
1. *Modern recipes use pre-LN* — GPT-2/3, LLaMA, all the open weights.
1. *StepCheck:* of {GPT-2, BERT-original, T5, LLaMA-2}, which uses post-LN? (Expected: `BERT-original`.)
1. *Build the block widget* — drag pieces; see ✓ pre-LN / ✓ post-LN / ✗ broken.
1. *Endgame callback.*
   **Widgets:** `lnPlacementGrad`, `buildTheBlock`.
   **Estimated minutes:** 14.

### Lesson 16.3 — The residual stream as the object

**Summary:** Stop thinking of the stream as plumbing and start thinking of it as the noun the model is operating on.
**Steps (12):**

1. *The reframe* — every block reads, computes, writes. The thing it's reading from is the same thing it's writing to.
1. *Decomposition by addition* — `h_N = h_0 + Σ Δ_ℓ`. The whole forward pass is a sum.
1. *Click a position, see a vector* — `residualStreamScope` exploration mode.
1. *StepCheck:* If a layer's Δ is the zero vector for a token, what is that block doing for that token? (Expected: `nothing` / scalar 0 contribution.)
1. *Direct path* — what happens with N=0. Bigram floor.
1. *`directPathInspector` widget — slide block budget from 0 to N* — watch the prediction sharpen.
1. *Bandwidth contention* — there are far more "computational dimensions" than `d`. Heads compete.
1. *StepCheck:* GPT-2-small has 12 heads of dim 64, FFN width 3072, `d=768`. How many "computational dimensions" want to write at each layer relative to `d`? (Expected: ≈ `4 · 768 / 768 = 4` from FFN alone.)
1. *Memory management* — some heads/neurons appear to delete information. Brief callout to Elhage et al. 2021.
1. *Why this matters for m17 onwards* — interpretability lives here.
1. *Logit lens preview* — at any layer, project by `W_E^T` to see "current best guess."
1. *Endgame callback.*
   **Widgets:** `residualStreamScope`, `directPathInspector`.
   **Estimated minutes:** 18.

### Lesson 16.4 — Stacking N and the full forward pass

**Summary:** Stack the block N times, add a final LN and unembedding, and you have GPT.
**Steps (13):**

1. *Stack the block.*
1. *The final LayerNorm* — required precisely *because* pre-LN never normalized the trunk.
1. *Unembedding `W_U`.*
1. *Weight tying* — `W_U = W_E^T`. Press & Wolf 2017.
1. *Why tying is also a regularizer.*
1. *StepCheck:* In a tied character model with `V = 65, d = 64`, how many params are in {embedding ∪ unembedding}? (Expected: `4160`.)
1. *Softmax over vocab* — m14 callback; this is the bigram-loss machinery.
1. *Tracing the full forward pass* — animation: token IDs → embedding → +position → block 1 → … → block N → final LN → `W_U` → softmax.
1. *Depth specialization* — early/middle/late blocks tend to do different work. Empirical, not architectural.
1. *Run an ablation* — `ablationLab`; kill one FFN, watch NLL bump.
1. *StepCheck:* For a tiny 4-block model, ablating which one block usually hurts most? (Expected: `last` / final block — it directly shapes logits.)
1. *Stack N for any N* — same block, repeated.
1. *Endgame callback.*
   **Widgets:** `ablationLab`, plus the existing `residualStreamScope`.
   **Estimated minutes:** 16.

### Lesson 16.5 — Counting the cost

**Summary:** Where does GPT's parameter and FLOP budget actually go?
**Steps (10):**

1. *Per-block parameter count* — `4d² + 8d² = 12d²`.
1. *Why FFN dominates as `d` grows* — both terms are `d²`, FFN's coefficient is bigger.
1. *StepCheck:* Per-block parameters at `d=768`. (Expected: `~7.08M` (or `12 · 768² = 7,077,888`).)
1. *Full model: GPT-2 small derivation* — line by line, end at ~124M.
1. *StepCheck:* Total params for GPT-2 small with weight tying. (Expected: `~124M`.)
1. *FLOPs per token* — `~6N` rule of thumb (Kaplan).
1. *The `T²` term* — when does it dominate? (`T > 8d`, roughly.)
1. *`paramBudgetPie` open exploration* — drag every knob; spot the regime changes.
1. *What we are about to do in m17* — sample from this thing.
1. *Endgame callback.*
   **Widgets:** `paramBudgetPie`.
   **Estimated minutes:** 12.

## 7. Problem bank

1. **(novice / computation / `four-x-expansion`)** Given `d_model = 1024`, what is the standard `d_ff`? **Answer:** `4096`.
1. **(novice / interpretation / `pre-ln-block`)** In the pre-LN block, does the residual trunk `x` ever pass through a LayerNorm between blocks `ℓ` and `ℓ+1`? **Answer:** No. (The trunk is only normalized once, at the top of the stack, by the final LN.)
1. **(novice / computation / `block-parameter-budget`)** Per-block parameters (no biases) for `d=512`. **Answer:** `12 · 512² = 3,145,728 ≈ 3.15M`.
1. **(novice / construction / `position-wise-ffn`)** Write the FFN as a function of one token's vector `z ∈ ℝ^d`, using GELU. **Answer:** `FFN(z) = GELU(z W_1 + b_1) W_2 + b_2`, with `W_1 ∈ ℝ^{d × 4d}`, `W_2 ∈ ℝ^{4d × d}`.
1. **(novice / interpretation / `weight-tying`)** What does it mean for an LM to have "tied embeddings"? **Answer:** The output projection (LM head) `W_U` equals `W_E^T`, where `W_E` is the input embedding matrix.
1. **(intermediate / computation / `model-parameter-budget`)** Estimate total tied params for `V=50257, d=768, N=12, T_max=1024, d_ff=4d`. **Answer:** `V·d + T·d + 12·N·d² + 2d ≈ 38.6M + 0.79M + 84.93M ≈ 124.3M`.
1. **(intermediate / interpretation / `residual-stream`)** Express `h_N` (residual stream after block N) in terms of `h_0` and the per-block contributions `Δ_ℓ`. **Answer:** `h_N = h_0 + Σ_{ℓ=1}^{N} Δ_ℓ`, where each `Δ_ℓ = MHA_ℓ(LN(h_{ℓ-1})) + FFN_ℓ(LN(h_{ℓ-1} + MHA_ℓ(LN(h_{ℓ-1}))))` (i.e., the sum of the two sub-layer outputs in block `ℓ`).
1. **(intermediate / computation / `block-flops`)** With `d=512, T=1024`, FFN forward FLOPs per block per sequence (multiply-add × 2). **Answer:** `2 · 2 · T · d · 4d = 16 · 1024 · 512² ≈ 4.29 GFLOPs`.
1. **(intermediate / debugging / `pre-vs-post-ln-gradient`)** A learner trains a 24-layer post-LN model with constant LR 3e-4 and reports immediate divergence. Suggest two architectural changes (in order of preference) without changing depth. **Answer:** (1) Switch to pre-LN. (2) If staying with post-LN, add a linear LR warmup of ~4000 steps.
1. **(intermediate / interpretation / `bigram-as-zero-layer`)** With weight tying, position embeddings zeroed, and `N=0`, what does the forward pass compute on input token `t`? **Answer:** `softmax(W_E[t] · W_E^T)` — i.e., the bigram next-token distribution from m14.
1. **(intermediate / construction / `full-forward-pass`)** Write the forward pass as a sequence of named operations, top to bottom, for input token IDs `t_{1:T}`. **Answer:** `t → E[t] + P[1:T] → block_1 → ... → block_N → LN_final → · W_U → softmax`.
1. **(intermediate / interpretation / `pre-ln-block` vs `post-ln-block`)** Which architecture requires a final LayerNorm at the top of the stack and why? **Answer:** Pre-LN, because the residual trunk is never normalized inside the stack; without the final LN, the activations entering `W_U` have unbounded scale.
1. **(intermediate / computation / `block-parameter-budget`)** What fraction of a transformer block's non-LN params live in the FFN? **Answer:** `8d² / (4d² + 8d²) = 2/3`.
1. **(advanced / proof-scaffold / `pre-vs-post-ln-gradient`)** Sketch why pre-LN bounds the input-output Jacobian norm of one block by `1 + ‖∂Sub/∂x‖`, while post-LN's depends on the LN's input-dependent scaling. **Answer:** For pre-LN, `y = x + S(LN(x))`, so `∂y/∂x = I + S'·LN'`. The `I` makes the operator norm `≥ 1` and prevents vanishing; the additive structure means stacking `L` blocks gives `Π (I + ε_ℓ) ≈ I + Σ ε_ℓ` — gradients route through the trunk. For post-LN, `y = LN(x + S(x))`, so `∂y/∂x = LN'(x + S(x)) · (I + S')`; the `LN'` factor scales by the input's standard deviation, and after the residual add, `‖x + S(x)‖` grows with depth — `LN'` shrinks, the output "forgets" `x`, and gradients to early layers attenuate.
1. **(advanced / computation / `model-parameter-budget`)** GPT-3 has `V≈50257, d=12288, N=96, d_ff=4d`. Estimate total params (no tying assumed for simplicity). **Answer:** Embedding: `V·d ≈ 0.62B`. Per block: `12 d² ≈ 1.81B`. Stack: `96 · 1.81B ≈ 173.7B`. LM head (untied): `V·d ≈ 0.62B`. Total ≈ `175B`.
1. **(advanced / debugging / `position-wise-ffn`)** A learner claims to "speed up the FFN" by averaging across the token dimension before the FFN and then broadcasting back. What breaks? **Answer:** The FFN is supposed to be position-wise — it must operate on each token's representation independently. Averaging across tokens makes every token's FFN output identical, destroying the position-specific information attention just routed there. The model collapses to producing the same hidden representation per position, modulo attention contribution — perplexity will explode.
1. **(advanced / interpretation / `residual-stream`)** Why does the "logit lens" — projecting an intermediate residual-stream vector through `W_U` — produce interpretable next-token guesses for GPT-2 but not always for other models? **Answer:** The logit lens assumes intermediate `h_ℓ` lives in the same coordinate system that the final LN + `W_U` decode from. In pre-LN GPT-2 with weight tying, this is approximately true because every block writes additively into a stable basis. In other models (GPT-Neo, BLOOM), intermediate layers can use a rotated/shifted basis; the unembedding can't decode them correctly. The "tuned lens" (Belrose et al. 2023) fixes this with a learned per-layer affine probe.
1. **(advanced / construction / `stacked-blocks` + `final-layer-norm` + `unembedding`)** In ~6 lines of pseudocode, write the full forward pass of a pre-LN, weight-tied GPT given `tokens, blocks[N], W_E, W_pos, gamma_f, beta_f`. **Answer:**

   ```
   h = W_E[tokens] + W_pos[: len(tokens)]
   for block in blocks:
       h = h + block.mha(block.ln1(h))
       h = h + block.ffn(block.ln2(h))
   z = layernorm(h, gamma_f, beta_f)
   logits = z @ W_E.T
   return softmax(logits, axis=-1)
   ```
1. **(advanced / interpretation / `block-parameter-budget`)** A user proposes "saving params" by halving the FFN width to `2d`. By what fraction does the block shrink, and what is the typical perplexity cost in the literature? **Answer:** Block params drop from `12d² → 8d²`, a 33% reduction. Empirically, halving `d_ff` from `4d` to `2d` typically costs a small but non-trivial perplexity bump (~1–3% relative), recoverable by training longer or adding depth. Modern gated FFNs (SwiGLU) actually reduce `r` to ~8/3 because the gating doubles the matrix count.
1. **(advanced / proof-scaffold / `weight-tying`)** Show that with weight tying, the gradient of the NLL with respect to a single embedding row `W_E[t]` receives contributions both from positions where `t` appears as input *and* positions where `t` is the target. **Answer:** Loss `L = -Σ_i log p(t_{i+1} | t_{≤i})`. The probability `p` depends on `W_E[t]` through (a) the embedding lookup at any input position equal to `t` (contribution flows through every block back to the row), and (b) the LM head row used to score `t` as a candidate target at every position (contribution is `(p(t|·) − 1[t=target]) · z_i` for every position `i`). Without tying, these are two different parameter tensors; with tying, both gradients accumulate into the same row, biasing it toward the output role (cf. recent analyses suggesting tied embeddings drift toward an output-friendly geometry).

## 8. Endgame callback — refined

The user's starter is fine but long. Three options, terser and landed for an adult who has earned the moment:

1. **"Tokens at the bottom. Attention and MLPs in the middle. Unembedding at the top. Bigram NLL on the outside. Stack N, train with Adam, warm up the LR. That is GPT — there is nothing else."**
1. **"You just built the noun. Stack N of these blocks, point Adam at the bigram loss from m14, and the only thing left is to choose how to sample. That's m17."**
1. **"Every modern LLM is this picture, repeated. The rest is engineering: tokenizer, optimizer, sampler, scale."**

Recommended: **option 1.** It explicitly names every primitive the course built up to and ends with the empirical reality ("there is nothing else") — which is what an adult learner who just spent five modules getting here actually wants to hear.

## 9. Sources (licensing-aware)

1. **"On Layer Normalization in the Transformer Architecture"** — Xiong, Yang, He et al., 2020. arXiv 2002.04745. Paper. arXiv preprint, copyright authors. **[REFERENCE-ONLY]** — use for the pre-LN gradient argument and the mean-field theorem; do not reproduce figures. This is the canonical citation behind Lesson 16.2.
1. **"A Mathematical Framework for Transformer Circuits"** — Elhage, Nanda, Olsson et al., Anthropic, 2021. https://transformer-circuits.pub/2021/framework/index.html. Blog/paper. transformer-circuits.pub does not display an open license; default copyright presumed. **[REFERENCE-ONLY]** — use for the residual-stream-as-object framing, bandwidth-contention argument, and direct-path / virtual-weights ideas underpinning Lessons 16.3 and 16.4. Cite, paraphrase, do not copy figures.
1. **"Attention Is All You Need"** — Vaswani et al., 2017. arXiv 1706.03762. Paper. The arXiv version notes "Google hereby grants permission to reproduce the tables and figures in this paper solely for use in journalistic or scholarly works"; that does not extend to a paid commercial product. **[REFERENCE-ONLY]** — use for the original post-LN block, dropout placement, the 4× FFN convention, and as the post-LN baseline in Lesson 16.2. The math itself is uncopyrightable; do not reproduce figures.
1. **"Using the Output Embedding to Improve Language Models"** — Press & Wolf, 2017. arXiv 1608.05859 / EACL. Paper. **[REFERENCE-ONLY]** — the canonical weight-tying citation for Lesson 16.4. Math is uncopyrightable; the perplexity-improvement claim is the citation hook.
1. **"Gaussian Error Linear Units (GELUs)"** — Hendrycks & Gimpel, 2016. arXiv 1606.08415. Paper, plus reference code at github.com/hendrycks/GELUs. **[REFERENCE-ONLY]** for the paper text/figures; the paper's *math* (the exact and tanh-approximate forms) is uncopyrightable and may be reproduced as formulas. The reference repository has no LICENSE file as of recent inspection — **treat code as [REFERENCE-ONLY]**.
1. **nanoGPT** — Karpathy. https://github.com/karpathy/nanoGPT. Code (Python). **MIT License.** **[ADAPT]** — `model.py` is the canonical compact implementation of a pre-LN GPT-2 block; suitable to adapt with attribution for code references in Lessons 16.1, 16.4, 16.5. The accompanying YouTube lectures ("Let's build GPT", "Let's reproduce GPT-2") are standard YouTube license — **[REFERENCE-ONLY]**.
1. **picoGPT** — Jay Mody. https://github.com/jaymody/picoGPT. Code. The accompanying blog post "GPT in 60 Lines of NumPy" (https://jaykmody.com/blog/gpt-from-scratch/) explicitly walks through every formula for a forward pass. The repository has an MIT LICENSE in its root. **[ADAPT]** for the code; **[REFERENCE-ONLY]** for the blog prose.
1. **Transformer Explainer** — Cho, Kim, Karpekov et al., Georgia Tech (Polo Chau). https://poloclub.github.io/transformer-explainer/. Code repo at https://github.com/poloclub/transformer-explainer. Software is **MIT-licensed**; the paper text/figures are arXiv preprint (2408.04619) so **[REFERENCE-ONLY]** for paper text. **[ADAPT]** the code/UI patterns (with attribution) for widget design — particularly the "abstraction-level transitions" pattern relevant to `residualStreamScope`.
1. **LLM Visualization** — Brendan Bycroft. https://bbycroft.net/llm. Code at https://github.com/bbycroft/llm-viz. The code is published as a repository but **the LICENSE was not clearly visible at the time of research** — verify directly before using. Treat as **[REFERENCE-ONLY]** unless and until the license is confirmed permissive. Use as design inspiration for `residualStreamScope` and `directPathInspector`.
1. **The Annotated Transformer** — Sasha Rush, Harvard NLP. https://nlp.seas.harvard.edu/annotated-transformer/. Paper ACL Anthology W18-2509 (CC BY 4.0 via ACL). Code at https://github.com/harvardnlp/annotated-transformer — **MIT License** at last check. **[ADAPT]** for the code and (with attribution under CC BY 4.0) for prose; **[REFERENCE-ONLY]** for derivative figures.

(Deliberately not recommended for adaptation: BertViz — Apache-2.0 code is technically usable but the design is attention-pattern-centric, redundant after m15; CircuitsVis — MIT, but Python/React-coupled in a way that doesn't fit Astro/Svelte; Wikipedia — CC BY-SA, reference-only; MIT OCW — CC BY-NC-SA, reference-only; Khan Academy — CC BY-NC-SA, reference-only; 3Blue1Brown videos — standard YouTube license, reference-only.)

## 10. Pedagogical traps

1. **Trap: Re-explaining attention or LN inside this module.**
   *Why it happens:* The instinct is to make the lesson "self-contained," and the assembly module touches every primitive.
   *Mitigation:* Be ruthless. m15 owns attention; m13 owns LN/residual. Open every reference with a "from m15" / "from m13" callback link, and keep the exposition here strictly about *composition*. Cut anything that re-derives a primitive — link to the prior module's widget instead.
1. **Trap: Letting the FFN feel like a footnote.**
   *Why it happens:* Most popular tutorials (Illustrated Transformer, blog posts) lavish attention on attention and treat the FFN as a postscript. Students arrive with that prior.
   *Mitigation:* Open Lesson 16.5 with `paramBudgetPie` *before* the math. Make the user drag `d_model` and watch FFN's slice eat the screen. Once they've felt the 2/3 figure with their hands, the misconception is dead.
1. **Trap: Hand-waving the pre-LN argument.**
   *Why it happens:* The Xiong proof is mean-field theory; the empirical claim "post-LN needs warmup" is itself easy to assert without grounding.
   *Mitigation:* Two grounded artifacts: (a) the `lnPlacementGrad` widget shows the gradient norm curves at `L=24` at step 0 — visceral. (b) Cite Xiong et al. 2020 specifically for the theorem. Don't use the words "more stable" without showing the gradient norms diverge.
1. **Trap: Drowning in interpretability nuance.**
   *Why it happens:* The residual-stream framing is enormously rich (circuits, induction heads, superposition, the logit lens). It's tempting to teach all of it.
   *Mitigation:* Lesson 16.3's job is to install the *frame*, not the entire interpretability literature. One widget (`residualStreamScope`), one decomposition rule (sum of Δ's), one preview (logit lens), and stop. Mech-interp can be its own module later.
1. **Trap: Drag-and-drop "build the block" widget being either too rigid or too permissive.**
   *Why it happens:* If the validator only accepts one canonical wiring, students can't explore. If it accepts everything that runs, they don't learn what's "broken."
   *Mitigation:* Define exactly three labels — *valid pre-LN*, *valid post-LN*, *broken* — and show *which property* failed (e.g., "the residual trunk passes through an LN; this is post-LN, not broken" vs. "no residual connection on the second sub-layer; this won't train at depth"). Make the failure modes pedagogical, not just binary.
1. **Trap: Treating parameter-counting as bookkeeping.**
   *Why it happens:* It looks like arithmetic. Adult learners already used to back-of-envelope dimensional analysis can zone out.
   *Mitigation:* Frame the parameter pie as the *answer to a why question*: "Why does GPT-2 small have ~124M params and not 50M or 500M?" Show the arithmetic *as a derivation* of an actually-deployed model (GPT-2 small with verified numbers), not a generic formula. Then snap-to-preset GPT-2 medium / GPT-3 to make the scaling visceral.
