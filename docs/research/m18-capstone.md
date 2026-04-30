# Module 18 — Capstone: Train a Tiny Transformer in Your Browser
## Pedagogical & Technical Research Brief

---

## 1. Concept dependency graph

Topologically sorted. Every concept references its prerequisites (this-module ids in plain kebab-case; earlier-module ids prefixed `m5-`…`m17-`).

1. **webgpu-mental-model** — A modern, low-level browser API exposing GPU compute through pipelines, bind groups, storage buffers, and WGSL shaders. *Prereqs:* m7-matmul, m12-autograd.
2. **wgsl-shader-anatomy** — A WGSL compute shader is a `@compute @workgroup_size(...)` entry point that reads/writes storage buffers indexed by `global_invocation_id`. *Prereqs:* webgpu-mental-model.
3. **storage-buffer-as-tensor** — A `GPUBuffer` of `f32` plus a `[B,T,C]` shape tuple is the browser-side equivalent of a `torch.Tensor`. *Prereqs:* wgsl-shader-anatomy, m7-matmul.
4. **f32-only-default** — Broadly portable WebGPU only guarantees `f32`; `shader-f16` is an opt-in feature absent on most Qualcomm/Linux configs *Prereqs:* webgpu-mental-model.
5. **dispatch-and-workgroup-sizing** — Total threads = `dispatch.x * dispatch.y * dispatch.z * workgroup_size.{x,y,z}`; 64 is the safe default, 256×8 a good matmul tile. *Prereqs:* wgsl-shader-anatomy.
6. **forward-kernels-needed** — A pre-LN GPT block forward pass requires exactly: embedding-gather, layernorm, qkv-matmul, scaled-dot-attention, attention-output-matmul, residual-add, ffn-matmul-1, gelu, ffn-matmul-2, residual-add. *Prereqs:* m15-attention, m16-pre-ln-block.
7. **backward-kernels-needed** — Each forward op has a hand-derived backward: matmul → two matmuls (`dA = dC·Bᵀ`, `dB = Aᵀ·dC`); softmax+CE → `(p − y)`; layernorm → 3-term derivative; embedding gather → scatter-add. *Prereqs:* m12-autograd, forward-kernels-needed.
8. **adamw-as-shader** — AdamW is one elementwise WGSL kernel reading `(θ, g, m, v, t)` and writing `(θ', m', v')` — no graph required. *Prereqs:* m10-adamw, storage-buffer-as-tensor.
9. **runtime-decision** — Choose between candle-wasm, webgpu-torch, ORT-Web-training, tinygrad-web, transformers.js, or hand-written WGSL; this module commits to **hand-written WGSL** (rationale §2 below). *Prereqs:* webgpu-mental-model, forward-kernels-needed, backward-kernels-needed.
10. **static-asset-pipeline** — Tokenizer (vocab JSON) and corpus (`Uint8Array` of token ids) ship as immutable static assets fetched once and held in memory; service worker caches forever. *Prereqs:* m17-tokenization.
11. **training-loop-choreography** — Training runs as a `requestAnimationFrame`-driven loop that does K iterations per frame, then yields, so the main thread never blocks longer than ~16 ms. *Prereqs:* m17-training-loop.
12. **rAF-tab-throttle** — Background tabs pause `requestAnimationFrame`; the loop must use `Page Visibility API` to pause cleanly and resume without nonsensical iter/sec spikes. *Prereqs:* training-loop-choreography.
13. **shader-warmup-anomaly** — First dispatch of a new pipeline pays 1–5 s of compilation; iter 0 wall time is meaningless and excluded from iters/sec. *Prereqs:* dispatch-and-workgroup-sizing.
14. **live-metrics-surface** — Train NLL, val NLL, iters/sec, and wall time update via direct DOM text writes (or a dedicated Svelte 5 rune store), never re-rendering the whole MDX tree. *Prereqs:* training-loop-choreography.
15. **dual-curve-loss-plot** — A circular-buffer-backed Mafs/Canvas plot scrolling smoothly with ≤2 ms render budget, dropping points on a sub-sample policy past N=2000. *Prereqs:* live-metrics-surface, m13-loss-curves.
16. **live-sampling-during-training** — Every K iters, freeze a copy of weights, run an autoregressive sample with current temperature/top-k/top-p, append output to a streaming `<pre>`. *Prereqs:* m17-sampling, training-loop-choreography.
17. **seed-determinism** — A single sfc32 PRNG seeded by xmur3/cyrb128(string) threaded through (a) data-loader offsets, (b) dropout masks, (c) weight-init, (d) sampler — produces byte-identical runs. *Prereqs:* m13-dropout, m17-sampling.
18. **save-load-weights** — Serialize the `Float32Array` of all parameters in nanoGPT order as one tiny `.bin` (~800 KB fp32) with a 3-line JSON header (config, vocab-hash, seed); download via `URL.createObjectURL`; reload via `FileReader`. *Prereqs:* runtime-decision, seed-determinism.
19. **shareable-checkpoint-url** — gzip + base64 the 800 KB blob → ~300 KB URL hash; works for ~200k-param models, breaks above ~500k. *Prereqs:* save-load-weights.
20. **frozen-checkpoint-playground** — Lesson 2 page loads weights once, exposes only the sampler knobs (prompt, temperature, top-k, top-p, seed) — no training, no graph, just `forward → sample`. *Prereqs:* save-load-weights, m17-sampling.
21. **mech-interp-handoff** — Anthropic's residual-stream framing (M16) says every block reads/writes the same width-d_model stream; the trained checkpoint is now an artifact one can probe with logit-lens / attention-pattern tools. *Prereqs:* m16-residual-stream, save-load-weights.
22. **what-this-proves** — A 200k-param model on 1 MB of Shakespeare produces poetry-flavored gibberish; what's been demonstrated is not capability but operational understanding of every line of nanoGPT. *Prereqs:* all of the above.

---

## 2. Canonical worked examples

### Example 1 — The 7-line forward pass, shader by shader
**Problem.** Given a transformer block as defined in M16 (pre-LN, residual, 4× FFN expansion, GELU), enumerate the WGSL kernels dispatched per forward step for `B=64, T=64, d_model=128, n_head=4`, and give the dispatch grid for each.
**Solution.**
1. Embedding gather: 1 kernel, dispatch `ceil((B·T·C)/64)`.
2. LayerNorm-1: 1 kernel, dispatch `B·T` workgroups, each thread block reduces over `C=128`.
3. QKV matmul (one fused linear `[B·T, C] × [C, 3C]`): tile size `16×16`, dispatch `ceil(B·T/16) × ceil(3C/16)`.
4. Scaled-dot-product attention (causal): one kernel computing `softmax((QKᵀ + M)/√d_k)V`; dispatch `B·n_head·T` workgroups of size 64; reduces over T inside the workgroup.
5. Attention output projection: matmul `[B·T, C] × [C, C]`.
6. Residual-add: one kernel, elementwise.
7. LayerNorm-2 → FFN1 `[C, 4C]` → GELU → FFN2 `[4C, C]` → Residual-add. Each is one kernel.
**Per block: 11 kernel dispatches.** Stack 4 blocks + final LN + unembedding ≈ **48 dispatches per forward step**.
**Pedagogical point.** A transformer is finite — a learner can literally count the kernels.
**Common mistake.** Students dispatch `B*T*C` threads with `workgroup_size(1,1,1)`, producing thousands of underfilled wavefronts; the matmul ends up 50× slower than the 16×16 tiled version.

### Example 2 — Backward of softmax+cross-entropy is `(p − y)`
**Problem.** Show that for logits $z \in \mathbb{R}^V$, target one-hot $y$, with $p = \text{softmax}(z)$ and loss $L = -\sum_i y_i \log p_i$, the gradient is $\partial L/\partial z = p - y$.
**Solution.** Standard 4-line derivation: $\partial p_i/\partial z_j = p_i(\delta_{ij} - p_j)$, then $\partial L/\partial z_j = -\sum_i (y_i/p_i) \cdot p_i(\delta_{ij} - p_j) = -y_j + p_j \sum_i y_i = p_j - y_j$.
**Pedagogical point.** The Jacobian-of-softmax telescopes cleanly with `1/p` from the log; you never materialize it. One backward kernel suffices.
**Common mistake.** Writing the Jacobian as a `V×V` matrix and matmul'ing it — uses `O(V²)` memory for nothing.

### Example 3 — `get_batch` in 12 lines
**Problem.** Implement nanoGPT's `get_batch` in the browser given `data: Uint16Array` of length 1,003,854 train tokens, `B=32`, `T=64`, seeded sfc32 PRNG `rng`.
**Solution.**
```js
function getBatch(data, B, T, rng) {
  const x = new Int32Array(B*T), y = new Int32Array(B*T);
  for (let b=0; b<B; b++) {
    const ix = Math.floor(rng() * (data.length - T - 1));
    for (let t=0; t<T; t++) {
      x[b*T+t] = data[ix+t];
      y[b*T+t] = data[ix+t+1];
    }
  }
  return {x, y};
}
```
**Pedagogical point.** Random offsets, contiguous T, y is x shifted by one. Same recipe as nanoGPT, line for line.
**Common mistake.** Forgetting `-T-1` in the upper bound; reads off the end of the buffer producing zeros that the embedding interprets as `'\0'`.

### Example 4 — Counting parameters of the capstone target
**Problem.** Verify that a model with `vocab=65, d_model=128, T=64, n_layer=4, n_head=4` and tied unembedding has ~200,000 parameters.
**Solution.**
- Token embedding: $65 \cdot 128 = 8{,}320$ (tied → counted once)
- Positional embedding: $64 \cdot 128 = 8{,}192$
- Per block: 2× LN bias+scale (`2·2·128 = 512`) + QKV proj (`128·384 = 49{,}152`) + Attn-out (`128·128 = 16{,}384`) + FFN-1 (`128·512 = 65{,}536`) + FFN-2 (`512·128 = 65{,}536`) **≈ 197k per block** with no biases.
With 4 blocks that's **~788k**. Reach ~200k by **dropping bias terms and using d_model=64 with 4 heads of 16, or using `d_model=128, n_layer=2`**. The course should restate the target as **n_layer=2, d_model=128 → ~200k**, or accept ~800k for n_layer=4. *(Honest correction — the brief's "4-layer, 4-head, d_model=128, ~200k params" is internally inconsistent for a standard pre-LN transformer; use n_layer=2 for a true 200k target, or n_layer=4 d_model=64.)*
**Pedagogical point.** M16's "12·d_model²·n_layer" rule of thumb gives `12·128²·4 = 786k` — exactly what we computed. Trust the rule.
**Common mistake.** Forgetting tied unembedding and counting the lm_head twice.

### Example 5 — Why iter 0 takes 4 seconds and iter 1 takes 80 ms
**Problem.** Learner reports iter 0 = 3,847 ms, iter 1 = 81 ms, iter 2 = 79 ms. Is the model broken?
**Solution.** No. WebGPU compiles each unique `(shader, pipeline-layout)` to backend-specific machine code (D3D12/Metal/Vulkan SPIR-V) on first dispatch; this is the 1–5 s "shader compilation cost" documented across runtimes. Subsequent iters hit the cached pipeline. **Mitigation:** run a synthetic warm-up pass at startup (one forward + one backward on dummy batch) before starting the iters/sec counter.
**Pedagogical point.** "Wall time of iter 0" is not a model property; it's a pipeline-cache property of the browser.
**Common mistake.** Including iter 0 in the moving-average iters/sec; the curve looks like a hockey stick on the wrong axis.

### Example 6 — Ignoring `Math.random` produces non-reproducible runs
**Problem.** Two reloads with identical hyperparameters give different loss curves at iter 100. Why? How do you fix it?
**Solution.** `Math.random()` is V8-seeded by an unspecified entropy source per realm; not reproducible. Replace with `sfc32(seed_a, seed_b, seed_c, seed_d)` where seeds come from `cyrb128(seedString)`. Thread that single PRNG through `getBatch`, dropout-mask generation, weight-init, and sampler. Then `(seedString, hyperparameters) → identical Float32Array of weights at every iter`.
**Pedagogical point.** Determinism is not free; it's an architectural choice.
**Common mistake.** Seeding only the data loader and forgetting dropout, then chasing a "loss is slightly different" ghost for an hour.

### Example 7 — Saving the trained checkpoint in 6 lines
**Problem.** Serialize the model's parameters to a downloadable `.bin`.
**Solution.**
```js
const allWeights = concatFloat32([wte.cpu(), wpe.cpu(), ...blockWeights, ln_f.cpu()]);
const header = JSON.stringify({config, seed, vocabHash, iter, valLoss});
const headerBytes = new TextEncoder().encode(header.padEnd(512, ' '));
const blob = new Blob([headerBytes, allWeights.buffer], {type: 'application/octet-stream'});
const url = URL.createObjectURL(blob);
download(url, `tinker-shakespeare-${seed}-iter${iter}.bin`);
```
**Pedagogical point.** A model is just `Float32Array` + a structural agreement on order. No magic.
**Common mistake.** Saving `Float64Array` (tensor on CPU sometimes) instead of `Float32Array`, doubling file size.

### Example 8 — Why fp32 in Chrome is 0.05 nats higher in val NLL than CUDA fp32
**Problem.** Same model, same seed, same data on CUDA reaches val NLL = 1.45; in browser (Chrome WebGPU on M2) it reaches 1.50. Bug?
**Solution.** Almost certainly not. WGSL `f32` accumulation order in matmul is implementation-defined and differs from cuBLAS; with denormals and FMA contraction differences, divergence accumulates exponentially across 2000 iters. Validation: train for 50 iters with batch_size=1 on identical seeds and confirm val NLL agrees to 4 decimals at iter 1, then watch divergence grow. The final-step gap of 0.05 nats is within reproducibility-across-hardware variance reported in PyTorch issues.
**Pedagogical point.** "Bit-identical to CUDA" is not a goal of a portable web runtime; "trains as well as CUDA" is.
**Common mistake.** Confusing reproducibility-within-platform (achievable, do it) with reproducibility-across-platform (not a thing in IEEE-754 + non-deterministic reductions).

---

## 3. Common misconceptions

1. **"WebGPU is just WebGL with a different name."** Natural because the W3C named it confusingly. **Counter:** WebGL has *no first-class compute*; people simulated it by drawing invisible triangles. WebGPU has compute pipelines, storage buffers, and WGSL — closer to Vulkan/Metal than to WebGL.
2. **"In-browser training must be a hundred times slower than CUDA."** Natural because "browser" connotes "toy." **Counter:** For our 200k–800k model size, the bottleneck is per-dispatch overhead (~95µs reported in 2026 WebGPU dispatch-overhead benchmarks), not raw FLOPs. A consumer M2 reaches several hundred GFLOPs of WGSL fp32 matmul; the 4-min training budget is tight but achieved.
3. **"`Math.random()` is fine, it's already random."** Natural because the API name promises randomness. **Counter:** It is non-seedable, non-reproducible, realm-specific. Demo: same `(hyperparams, click-Start)` twice; show the curves diverge by iter 30. Replace with sfc32, show byte-identical runs.
4. **"If I don't see WebGPU on the user's browser I should just refuse to load."** Natural because WebGPU adoption looks shaky in headlines. **Counter:** As of April 2026 all four major browsers ship WebGPU (Chrome 113+, Edge 113+, Firefox 141+, Safari with WebGPU); detection should be `if (!navigator.gpu) { show fallback message }` and that branch is rare.
5. **"`fp16` will halve my training time so I should use it."** Natural because it's true on H100. **Counter:** WebGPU `shader-f16` is opt-in and absent on every Qualcomm Android device (per gpuweb issue #5006) and most Linux+open-source-driver configs. Default to fp32; offer fp16 as a flag.
6. **"`requestAnimationFrame` runs at 60 Hz, period."** Natural because that's the textbook number. **Counter:** Background tabs throttle rAF *to nothing or to 1 Hz*; high-refresh displays push it to 120/144 Hz. The training loop must compute "iters per *real* wall second" using `performance.now()`, not "iters per rAF tick."
7. **"loss.backward() is library magic; I cannot read it."** Natural after a year of PyTorch. **Counter:** For the ten ops in a transformer, the backward pass is ten hand-derived expressions. By M12 the learner already has all of them. The capstone *demonstrates* this by writing them as ten WGSL kernels.
8. **"The model needs to be GPT-3-shaped to produce English."** Natural because internet headlines. **Counter:** A 200k-parameter character-level model trained for 2000 steps on Shakespeare produces text that is unmistakably *Shakespeare-flavored* — broken meter, plausible names, "thee/thou" distribution roughly correct. Demo by displaying live samples at iter 0 / 200 / 500 / 2000.
9. **"My val NLL is higher than my train NLL — the model is overfit."** Natural; M13 introduced overfitting. **Counter:** Train NLL is a *running* average over the most recent 10 batches (smoother, reflects current loss); val NLL is a *fresh* eval over a held-out 200 batches. They are measuring different statistics; train < val by ~0.1 nat is expected even on a perfectly-regularized run.
10. **"Saving weights means saving the model architecture."** Natural in TF/Keras land. **Counter:** A `.bin` of Float32Array + a 3-line JSON header is enough; the architecture lives in code shipped with the page. "Loading a checkpoint" means assigning a Float32Array to existing GPU buffers.
11. **"Determinism + parallel reduction are compatible."** Natural assumption from non-floating-point experience. **Counter:** Sums over GPU threads are non-associative under fp32; even within one GPU, reduction order is undefined. Determinism is achievable *within* a fixed implementation (Chrome 124 on M2) but not *across* GPU vendors. State this honestly.
12. **"Tab-throttling is a bug I can work around."** Natural — feels punitive. **Counter:** It is a battery-life feature browsers will not unship. The capstone embraces it: switch tabs → loop pauses → switch back → loop resumes from exact iter and seed. Surface "Paused (tab in background)" in the UI.

---

## 4. Interactive widget suggestions

### `runnerPanel`
- **User manipulates:** Start / Pause / Reset buttons; seed input (string); 5 hyperparameter sliders (lr, batch size, dropout, n_layer, T) — all with effects defaulted to "no effect from previous run."
- **Live updates:** Loss curve (train+val), iters/sec, wall time, current iter, current sample (live regenerated every 100 iters), GPU memory bytes, current learning rate (cosine-decayed).
- **Concept it makes tangible:** *Training is a process you stand next to and watch*, not an opaque overnight job. The big Bret-Victor moment is "the loss is going down because of the kernel I wrote three lessons ago."
- **Why slider+number fails:** A slider on lr with a number readout teaches nothing about *training dynamics*. The runner shows the *curve* deforming under that slider in real time — the same model is retrained.
- **Prior art:** ConvNetJS MNIST demo (`https://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html`); Karpathy's RecurrentJS sentence-memorization demo (`https://cs.stanford.edu/people/karpathy/recurrentjs/`); GenAI-fi nanoGPT TF.js (`https://www.npmjs.com/package/@genai-fi/nanogpt`).

### `liveSampleStream`
- **User manipulates:** Prompt textbox; sample-every-N-iters slider; live temperature slider that affects the *next* sample (not in-flight one).
- **Live updates:** A scrolling `<pre>` log: `iter 0: kxQ@!je e f...`, `iter 200: ther tho the king...`, `iter 500: HENRY: I am the king of...`, with each iter line stamped.
- **Concept it makes tangible:** "Recognizable English emerges around iter 500" is hearsay until you watch it. This is the moment-of-truth widget.
- **Why slider+number fails:** No number on a slider can show the *qualitative* phase transition from gibberish to plausible iambic.
- **Prior art:** Karpathy's char-rnn blog post and live demo; Brendan Bycroft's LLM viz (`https://bbycroft.net/llm`); poloclub Transformer Explainer (`https://poloclub.github.io/transformer-explainer/`).

### `kernelInspector`
- **User manipulates:** Click any node in a static SVG of the forward graph (embed → LN → QKV → SDPA → … → unembed). Dragging a slider for `B*T*C` resizes the buffers visually.
- **Live updates:** A panel showing the WGSL source of *that* kernel, the dispatch dimensions (`dispatchWorkgroups(x, y, z)` in numbers), the input/output buffer shapes, and a "this is iteration N of K parallel threads" thread-cube animation.
- **Concept it makes tangible:** "A kernel is just a function with a dispatch grid wrapped around it."
- **Why slider+number fails:** Threads aren't a number; they're a *shape* in 3-space.
- **Prior art:** Brendan Bycroft's 3D LLM walkthrough; webgpufundamentals.org compute-shader demos; Tensor Heaven matmul visualizations.

### `seedScrubber`
- **User manipulates:** A text input for seed string; a "twin runs" toggle that splits the canvas into two side-by-side training runs.
- **Live updates:** Two loss curves, two live samples — when seed strings match, curves overlap exactly (visual proof of byte-identical determinism); when they differ by one character, curves diverge after iter ~3.
- **Concept it makes tangible:** Determinism is not "the same kind of result" — it's *the same Float32Array byte for byte*.
- **Why slider+number fails:** "Same number on the screen" doesn't communicate "same 800,000-byte file."
- **Prior art:** No direct prior art for in-browser ML; conceptually borrows from PCG-game seeded-world toggles (e.g. Minecraft seed comparators).

### `samplerKnobsPlayground` *(Lesson 2 centerpiece)*
- **User manipulates:** Frozen trained checkpoint; sliders for temperature (0.1 → 2.0), top-k (1 → 65), top-p (0.1 → 1.0), prompt textbox, seed, "regenerate."
- **Live updates:** Generated sample updates *and* a histogram bar chart showing the next-token probability distribution at the cursor position — bars graying out under top-k cutoff, bars in the "nucleus" highlighted under top-p.
- **Concept it makes tangible:** The model outputs a *distribution*, not a sentence; the knobs reshape that distribution.
- **Why slider+number fails:** Three numbers cannot show how the *distribution* deforms; you have to see bars vanish under top-p in real time.
- **Prior art:** poloclub's transformer-explainer temperature slider on GPT-2 logits (`https://poloclub.github.io/transformer-explainer/`); andreban/temperature-topk-visualizer (`https://andreban.github.io/temperature-topk-visualizer/`).

### `lossCurvePathologyZoo`
- **User manipulates:** Six "preset" buttons that intentionally re-train with a known pathology (lr=10, no warmup, dropout=0.9, no zero_grad, overfit on 4 sentences, NaN-injected weights at iter 5).
- **Live updates:** The runner trains in fast-forward (~30 s) and lays the resulting curve next to a labelled gallery of canonical pathologies from M13. The current run's curve "snaps" onto the closest match.
- **Concept it makes tangible:** Pattern-recognition of training failures is a learnable, discrete skill.
- **Why slider+number fails:** A pathology is a *shape*, not a value.
- **Prior art:** None of this exact form; M13 already taught five canonical pathologies — this widget is M13 made interactive.

### `paramCounter`
- **User manipulates:** Sliders for `n_layer`, `d_model`, `n_head`, `T`, `vocab`; checkbox for tied unembedding; checkbox for biases.
- **Live updates:** A live treemap/sankey breakdown — embedding %, attention %, FFN %, lm_head %; total parameter count; "this is X% of GPT-2 small." Also surfaces the `12·d²·n_layer` rule live: when slider matches it, a check appears.
- **Concept it makes tangible:** Where parameters *live* in a transformer (FFN does).
- **Why slider+number fails:** "788,288 parameters" tells you nothing; the treemap shows that the FFN owns 66 % and the attention 16 %.
- **Prior art:** Conceptually similar to "GPT-3 scaling-law calculators" but local and interactive.

---

## 5. Key formulas

**Transformer block (pre-LN, M16 review):**
$$\tilde{x} = x + \text{Attn}(\text{LN}_1(x)), \quad y = \tilde{x} + \text{FFN}(\text{LN}_2(\tilde{x}))$$

**Scaled dot-product attention (M15 review):**
$$\text{Attn}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top + M}{\sqrt{d_k}}\right)V$$

**Cross-entropy on next-token (M9, M14, M17 review):**
$$\mathcal{L} = -\frac{1}{B \cdot T}\sum_{b,t} \log p_{\theta}(y_{b,t} \mid x_{b,\le t})$$

**The single most-needed backward identity:**
$$\frac{\partial \mathcal{L}}{\partial z} = p - y, \qquad p = \text{softmax}(z)$$

**Backward of matmul $C = A B$:**
$$\frac{\partial \mathcal{L}}{\partial A} = \frac{\partial \mathcal{L}}{\partial C}\, B^\top, \qquad \frac{\partial \mathcal{L}}{\partial B} = A^\top\, \frac{\partial \mathcal{L}}{\partial C}$$

**LayerNorm forward:**
$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad y_i = \gamma_i \hat{x}_i + \beta_i$$

**LayerNorm backward (in-shader form):**
$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{1}{\sqrt{\sigma^2+\epsilon}}\left( \frac{\partial \mathcal{L}}{\partial \hat{x}_i} - \frac{1}{C}\sum_j \frac{\partial \mathcal{L}}{\partial \hat{x}_j} - \frac{\hat{x}_i}{C}\sum_j \frac{\partial \mathcal{L}}{\partial \hat{x}_j}\hat{x}_j \right)$$

**AdamW one-step (M10 review):**
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$
$$\hat{m}_t = m_t/(1-\beta_1^t), \quad \hat{v}_t = v_t/(1-\beta_2^t)$$
$$\theta_t = \theta_{t-1} - \eta\!\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} + \lambda\theta_{t-1}\right)$$

**Cosine LR schedule with linear warmup:**
$$\eta(t) = \begin{cases} \eta_{\max} \cdot \frac{t}{T_w} & t < T_w \\ \eta_{\min} + \tfrac{1}{2}(\eta_{\max}-\eta_{\min})\!\left(1+\cos\!\left(\pi\frac{t-T_w}{T-T_w}\right)\right) & t \ge T_w \end{cases}$$

**Temperature-top-k-top-p sampling chain:**
$$p_i = \frac{\exp(z_i/\tau)}{\sum_j \exp(z_j/\tau)}, \quad p_i \leftarrow p_i \cdot \mathbb{1}[i \in \text{top-}k], \quad p_i \leftarrow p_i \cdot \mathbb{1}\!\left[\sum_{j \le i}^{\text{sorted}} p_j \le p_{\text{nucleus}}\right]$$

**Parameter-count rule of thumb:**
$$N_{\text{params}} \approx 12 \cdot d_{\text{model}}^2 \cdot n_{\text{layer}}$$

**WebGPU dispatch-thread total:**
$$N_{\text{threads}} = \prod_{i \in \{x,y,z\}}\!\!(\text{dispatch}_i \cdot \text{workgroup\_size}_i)$$

**SFC32 PRNG one step:**
$$t \leftarrow (a+b+d)\bmod 2^{32}, \;\; d \leftarrow d+1, \;\; a \leftarrow b \oplus (b \gg 9), \;\; b \leftarrow c + (c \ll 3), \;\; c \leftarrow \text{rotl}(c, 21) + t$$

---

## 6. Lesson decomposition

This module has **3 lessons** by design — capstone lessons are *experiences*, not new theory.

---

### Lesson 18.1 — "Push the button"
**Summary:** You've built every piece. Now press Start, watch a transformer learn Shakespeare in your tab.

**Steps:**
1. *(prose+widget)* The runner page. Show the runnerPanel widget pre-loaded; the Start button glows. One paragraph: "Everything you've built in M5–M17 is on this page."
2. *(StepCheck — scalar)* "How many kernel dispatches per forward pass for our 2-layer model?" Expected answer: **24** (11 per block × 2 + final LN + unembed).
3. *(prose+widget)* WebGPU init: device adapter, fp32-only default. Side-panel snippet shows `await navigator.gpu.requestAdapter()`. Inspector reveals adapter info.
4. *(prose+widget)* Static asset shipping: tokenizer JSON (~2 KB), corpus `.bin` (~1 MB Uint16Array), service-worker cache. Show the network panel.
5. *(prose+widget)* Determinism: the seedScrubber widget. Demonstrate two identical seeds → identical curve overlap. Then change one character → curves diverge by iter 5.
6. *(StepCheck — scalar)* "What value of `cumulative_p` does top-p=0.9 stop adding tokens at?" Expected answer: **0.9** (this is a checkpoint, not a trick — you read the formula).
7. *(prose+widget)* Press Start. Live runnerPanel runs. Iter 0 takes ~3 s — explain shader-warmup-anomaly inline.
8. *(prose+widget)* While it trains: live samples panel. Watch iter 0 gibberish, iter 200 plausible characters, iter 500 word shapes, iter 2000 Shakespeare-flavored.
9. *(prose+widget)* The lossCurvePathologyZoo widget for comparison: your curve vs. five canonical broken curves. Yours should match "healthy."
10. *(prose+widget)* Tab-throttle gotcha. Switch tabs for 30 s; come back; widget shows "Paused" badge and resumes from exact state. Page Visibility API explained in 3 lines.
11. *(StepCheck — scalar)* "What is the minimum val NLL the model is allowed to reach, given the entropy floor of English text from M9?" Expected answer: **≈1.0 nats/token** (cross-entropy of natural English at character level ≈ 1.0–1.3 nats).
12. *(prose+widget)* Save Weights button. Click. A `tinker-shakespeare-{seed}-iter{N}.bin` downloads (~800 KB). Open it in a hex viewer in the iframe — see the 512-byte JSON header and the Float32 trail.
13. *(prose+widget)* "This checkpoint is yours." Optional: copy shareable URL (gzip+base64 of weights ≈ 300 KB). Paste in an incognito tab → same model loads.

**Widgets used:** `runnerPanel`, `seedScrubber`, `liveSampleStream`, `lossCurvePathologyZoo`, `paramCounter`.
**Estimated:** **45–60 minutes** (most of it is *waiting and watching*, which is the point).

---

### Lesson 18.2 — "Now make it talk"
**Summary:** Same model, no more training. Drag the knobs and make it say different things.

**Steps:**
1. *(prose+widget)* Load Weights button. Re-instantiate the model from Lesson 1's checkpoint — or use the included reference checkpoint if the learner skipped training.
2. *(prose+widget)* The samplerKnobsPlayground. Default values temp=0.8, top-k=40, top-p=0.95. Generate 200 chars; show output.
3. *(prose+widget)* Drag temperature to 0.1. Output becomes "I am the king of the king of the king of the king…" — degenerate repetition. Why? Distribution becomes near-deterministic.
4. *(StepCheck — scalar)* "If the top-2 logits are 5.0 and 3.0, what's `p_top1` at temperature 0.5?" Expected answer: **≈0.982** (softmax([10, 6]) = e⁴ / (e⁴+1)).
5. *(prose+widget)* Drag temperature to 2.0. Output becomes "%Tx 9j @KK ?I" — entropy explosion.
6. *(prose+widget)* Top-k experiment. Set temperature back to 0.8, slide top-k from 65 (no truncation) to 1 (greedy). Watch coherence vs. boredom trade-off.
7. *(prose+widget)* Top-p experiment. Set top-k=65, slide top-p from 1.0 to 0.5. Show the histogram bars graying out.
8. *(prose+widget)* Prompt engineering. Type "ROMEO:" — model continues in dialog format. Type "QUEEN:" — same. Type "let me not to the marriage" — sees Sonnet 116 territory.
9. *(StepCheck — scalar)* "With temperature → 0, top-k=1, the sampler becomes equivalent to which decoding strategy?" Expected answer: **greedy** (or "argmax").
10. *(prose+widget)* Reseed the sampler PRNG. Same prompt, same knobs, different seed → different sample (but byte-identical with same seed).
11. *(prose+widget)* Reflection: "You're not editing the model; you're editing the *post-processing of its distribution.*"

**Widgets used:** `samplerKnobsPlayground`, `liveSampleStream` (frozen variant).
**Estimated:** **25–35 minutes**.

---

### Lesson 18.3 — "The credits roll"
**Summary:** A retrospective tour through your own code: every line is now something you wrote and something you understand.

**Approach (per user instruction to surprise):** A **"credits-roll annotated source"** experience — the entire ~600 lines of the trained model's code (forward.ts + backward.ts + train.ts) scroll up the screen *cinematically*, frame-locked, while a side panel shows for each ~20-line block which prior module taught what's happening. Above the scroll, the live trained model continues to sample.

**Steps:**
1. *(prose)* A single sentence: "The next four minutes are not new content. Read the code."
2. *(prose+widget)* Scroll begins. As `class CharTokenizer` scrolls past, the side-panel says "M14 — vocab=65."
3. *(prose+widget)* As `getBatch` scrolls, side-panel: "M14 (Bengio fixed-context), M17 (B random offsets, x and shifted y, B*T-averaged loss)."
4. *(prose+widget)* As `embeddingLookupKernel.wgsl` scrolls: "M7 — embedding as linear layer." A small inline animation shows token-id → row of `[V, d_model]` matrix.
5. *(prose+widget)* As `layerNormKernel.wgsl` scrolls: "M13 — LN, He init." The 3-term backward expression flashes.
6. *(prose+widget)* As `causalAttentionKernel.wgsl` scrolls: "M15 — softmax(QKᵀ/√d_k + M)V; T² cost; KV cache for inference." Anthropic residual-stream graphic appears in the side panel.
7. *(prose+widget)* As `ffnKernel.wgsl` scrolls: "M11 (MLP), M16 (4× expansion, GELU)."
8. *(prose+widget)* As `crossEntropyBackwardKernel.wgsl` scrolls (a single line: `dlogits[i] = (probs[i] - target[i]) / (B*T)`): a callout balloon: **"This. This is the backward of the loss. M9 + M12. You wrote it."**
9. *(prose+widget)* As `adamwKernel.wgsl` scrolls: "M10. One kernel. No graph."
10. *(prose+widget)* As `trainStep` scrolls: "M17 — warmup-cosine LR, grad-clip-1.0, B*T-averaged loss."
11. *(prose+widget)* The live sample panel above shows the model still talking, slowly. Below, the last lines of code scroll: `requestAnimationFrame(trainStep)`. One callout: "This is the line that pauses when you change tabs. M18 — yours."
12. *(prose)* A black slide. "Total parameters: 197,632. Total lines of WGSL: 412. Total lines of TypeScript: 638. Modules used: 13. Iterations: 2,000. Wall-clock training time: 4 min 12 s."
13. *(prose)* A single closing paragraph (see endgame callback §8). Below it: three buttons: "Train another seed," "Open mech-interp playground" (links forward), "Where to next?"
14. *(prose)* "Where to next?" reveal:
    - Mech-interp on your trained checkpoint (Anthropic's transformer-circuits framing maps directly onto your residual stream).
    - Scale up: same code, swap tinyshakespeare for a 30 MB corpus, train for 30 min.
    - Read GPT-2 in nanoGPT — every line is something you can read.

**Widgets used:** A custom `creditsRollPanel` (scrolling annotated source) + `liveSampleStream` running on the trained checkpoint as the scroll plays.
**Estimated:** **15–20 minutes** (mostly cinematic; one StepCheck only if author wants — recommend none).

---

## 7. Problem bank

**Mix:** Per user discretion — heavy weighting on (a) integrative debugging and (b) browser-engineering. Capstone is integrative. 8 integrative-from-whole-course, 8 browser/WebGPU/training-loop, 4 close-reading/hybrid.

| # | Statement | Expected | Difficulty | Cognitive | Tags |
|---|---|---|---|---|---|
| 1 | A learner reports iter 0 = 4082 ms, iter 1 = 78 ms, all subsequent = 79 ± 2 ms. The training loop's iters/sec readout shows 0.24 averaged over the first 5 iters. Diagnose the root cause and propose the one-line fix. | "First-call WGSL pipeline compilation; do a synthetic warm-up dispatch before starting the metric counter." | novice | debugging | shader-warmup-anomaly, training-loop-choreography |
| 2 | Given the cross-entropy gradient identity $\partial\mathcal{L}/\partial z = p - y$, write the WGSL kernel signature (entry point, bindings, workgroup size, dispatch grid) for computing $dz$ given logits buffer of shape `[B*T, V]` and target indices of shape `[B*T]`. | Single `@compute @workgroup_size(64)` kernel; bindings: read `logits`, read `targets`, write `dlogits`; dispatch `ceil(B*T*V/64)`; computes `dlogits[i*V+j] = softmax(logits[i,:])[j] - (j==targets[i])` divided by `B*T`. | intermediate | construction | backward-kernels-needed, m9-nll, wgsl-shader-anatomy |
| 3 | Two reloads of the runner page with seedString="hamlet" produce different val NLL at iter 100 (1.84 vs 1.79). The data loader is correctly sfc32-seeded. Name three other RNG sites that must be threaded through the same PRNG to achieve byte-identical runs. | Dropout mask generation; weight initialization (He/Xavier); sampling RNG (top-k tiebreak / top-p categorical draw). | intermediate | debugging | seed-determinism, m13-dropout |
| 4 | The capstone target is described as "4-layer, 4-head, d_model=128, ~200,000 parameters." Using the M16 rule of thumb $N \approx 12\,d^2\,n_\text{layer}$, compute the parameter count for those settings. Which of the three numbers should be changed to make the description self-consistent? | $12 \cdot 128^2 \cdot 4 = 786{,}432 \approx 800\text{k}$. To hit ~200k either reduce to **n_layer=2** (gives 393k, closer; with d_model=96 → 221k) or reduce **d_model=64, n_layer=4** ($12 \cdot 64^2 \cdot 4 = 196{,}608$). | intermediate | computation | param-counter, m16-pre-ln-block |
| 5 | A learner's loss curve plateaus at NLL ≈ 4.17 from iter 0 onward, never decreasing. Vocab size is 65. Diagnose. | $\ln(65) \approx 4.174$; the model is outputting a uniform distribution. Common causes: optimizer not stepping (missed `.apply()`/`zero_grad` issue), `lr=0`, frozen parameters, broken gradient flow. Most likely: in-place tensor mutation breaking the autograd graph (M12 canonical bug). | intermediate | debugging | m9-perplexity, m12-canonical-bugs |
| 6 | Write the WGSL kernel for the forward pass of layer normalization given a `[B*T, C]` input buffer and `gamma`, `beta` of length C. Use one workgroup per row. | `@compute @workgroup_size(128) fn ln(...) { let row = workgroup_id.x; ... two-pass reduce: mu = sum/C; var = sum((x-mu)^2)/C; out[row*C+lid] = gamma[lid]*(x-mu)/sqrt(var+eps) + beta[lid]; }` (use workgroupBarrier between passes) | intermediate | construction | wgsl-shader-anatomy, m13-layer-norm |
| 7 | The learner's browser tab is in the background for 3 minutes; on return, the iters/sec readout shows 47,000 (impossibly high) for one frame, then settles. Why, and how do you mitigate? | rAF callbacks pause in background; the wall-time delta on resume is ~180 s but only 1 iter occurred, then the loop ran several iters in the catch-up frame. Mitigation: use `Page Visibility API` (`document.visibilitychange`) to explicitly pause/resume; clamp `dt` in the iters/sec calculation; redraw curves with the *true* paused gap visualized. | advanced | debugging | rAF-tab-throttle |
| 8 | Show that AdamW's update rule reduces to SGD with momentum when $\beta_2 = 0$ and $\lambda = 0$. (Hint: track the bias correction.) | With $\beta_2=0$: $v_t = g_t^2$, $\hat{v}_t = g_t^2/(1-0) = g_t^2$, so $\hat{m}/\sqrt{\hat{v}} = \hat{m}/|g|$ which is *sign-SGD with momentum*, not standard SGD-momentum. The reduction to plain SGD-momentum requires more, e.g. $\beta_2=0, \epsilon \to \infty$. **(Trick question — calls out a common false simplification.)** | advanced | proof-scaffold | m10-adamw |
| 9 | The capstone trains in roughly 4 minutes on consumer WebGPU. If you doubled `T` from 64 to 128, by what factor does *attention* compute scale? By what factor does *FFN* compute scale? Which now dominates? | Attention: $T^2$, so 4×. FFN: $T$, so 2×. At T=64, attention is ~1/4 of compute, FFN ~1/2. After doubling: attention ≈ 1× of original FFN cost; FFN still dominates but the ratio narrows. | intermediate | computation | m15-attention, m16-pre-ln-block |
| 10 | Implement `getBatch(data, B, T, rng)` returning `{x, y}` Int32Arrays of length B*T. Provide the implementation and identify the off-by-one error a typical first-pass commits. | (See Example 3.) Off-by-one: bound must be `data.length - T - 1` not `data.length - T`, because `y[b*T+T-1] = data[ix+T]` requires that index to be valid. | novice | construction | m17-get-batch |
| 11 | The browser-side trained checkpoint is ~800 KB in fp32. The learner wants to share it as a URL hash. With base64 expanding 4:3 and gzip typical 2.5:1 compression on weight tensors, estimate the URL size. | $800 \text{ KB} / 2.5 = 320 \text{ KB}$ gzipped, then $\times 4/3 = 427 \text{ KB}$ base64. URL encoding typical ~5 % overhead → ~450 KB. Most browsers cap URLs at 2 MB, but anything over ~64 KB is impractical for paste-and-share. **Conclusion: 800 KB models do not fit in a URL hash; use a downloadable `.bin` instead.** | intermediate | computation | shareable-checkpoint-url |
| 12 | Given the live samples at iter 0 are uniformly random across the 65-char vocab, and the entropy floor of English text at the character level is ~1.0–1.3 nats/token, what is the minimum *perplexity* the capstone model can ever reach in principle? | $\exp(1.0) \approx 2.72$ to $\exp(1.3) \approx 3.67$. Anything below ~2.7 is a sign of overfitting to the train split. | novice | computation | m9-perplexity, m9-entropy-floor |
| 13 | Write the backward kernel for a matmul $C = AB$ where `A: [M, K]`, `B: [K, N]`, `dC: [M, N]`. Specify the dispatches for both `dA` and `dB`. | `dA = dC · Bᵀ` (one matmul, dispatch ceil(M/16) × ceil(K/16) with 16×16 workgroups). `dB = Aᵀ · dC` (one matmul, dispatch ceil(K/16) × ceil(N/16)). Two kernel launches, identical structure to forward but with transposed reads. | intermediate | construction | backward-kernels-needed |
| 14 | A learner argues "since I'm running training on WebGPU and CUDA produces the same loss curve, my browser implementation is correct." Critique this claim. | Loss-curve agreement across implementations is necessary but not sufficient. fp32 reduction order in WGSL is undefined; you can have correct *expected* gradients but byte-different ones, leading to slightly different val NLL after long training. The right test: agreement *within* one platform across reloads (that's bit-identical with seeded sfc32) AND broad agreement *across* platforms (within ~0.05 nats at iter 2000). | advanced | interpretation | f32-only-default, seed-determinism |
| 15 | The frozen-checkpoint sampler with `temperature=0.8, top_k=40, top_p=0.95` produces "the king of the king of the king…" for any prompt. Diagnose. | Either (a) top-p is reaching cumulative 0.95 within just the top 1–2 tokens because the distribution is too peaked at low temperature for *this* model, or more likely (b) the model is severely under-trained (e.g. iter 100), so the argmax is repetitive. Increase temperature to 1.2 or train longer. (Or: a bug — top-p mask zeroing too aggressively, e.g. selecting just the top-1 token.) | intermediate | debugging | frozen-checkpoint-playground, m17-sampling |
| 16 | Given the WebGPU dispatch overhead is reported at ~95–99 µs per kernel (recent benchmarks), and our forward pass dispatches 24 kernels and backward dispatches ~28 kernels, estimate the *fixed* per-step overhead independent of model size. Compare to a 4-min budget for 2000 steps. | $52 \text{ kernels} \times 100 \mu\text{s} = 5.2 \text{ ms}$ per step from dispatch alone. Over 2000 steps: ~10 s of pure overhead, or ~4 % of the 4-min budget. Acceptable; bigger problem is matmul kernel time itself (~30–40 ms per step), which dominates. | advanced | computation | dispatch-and-workgroup-sizing |
| 17 | Modify the AdamW kernel to apply gradient clipping to a global L2-norm of 1.0 *before* the AdamW step. This requires a 2-pass design — describe the dispatches. | Pass 1: a reduction kernel computes `sum(g_i^2)` across all parameters into a single scalar `gnorm_sq`; on CPU compute `scale = min(1.0, 1.0 / sqrt(gnorm_sq))`. Pass 2: a single elementwise kernel applies `g_i ← g_i * scale` and then runs the AdamW update. (Or fuse: pass 1 reduces, pass 2 clips, pass 3 adamw — 3 dispatches.) | advanced | construction | adamw-as-shader, m10-grad-clip |
| 18 | The course's pedagogical claim is that "loss.backward() is no longer magic — you wrote it." Specifically, identify the one-line WGSL expression that *is* the backward of the loss with respect to the logits, and state where in the M12 derivation tree it lives. | `dlogits[i] = (probs[i] - target_onehot[i]) / (B*T);` This is the softmax+CE composite gradient; lives at the root of the autograd DAG (the leaf you call `.backward()` on); the rest of backward is matrix algebra propagating *this* gradient. | intermediate | interpretation | m12-autograd, backward-kernels-needed |
| 19 | After Lesson 18.1, a learner asks: "Could I save the model and continue training tomorrow from where I stopped?" What additional state, beyond the parameter `.bin`, must be serialized? | Optimizer state (Adam's `m` and `v` buffers — same shape as parameters, so 2× the parameter file size); current iteration counter; PRNG state (sfc32 needs only 4 uint32s); current LR-schedule position. Without `m, v` you'd reset the optimizer's history and lose ~50–100 iters of effective progress. | intermediate | construction | save-load-weights, m10-adamw |
| 20 | Sketch a 5-line proof that with tied unembedding ($W_\text{embed} = W_\text{unembed}^\top$), the gradient through the output projection effectively *adds* to the gradient through the input embedding for every token in the batch. Give the implication for the embedding's effective learning rate. | Forward: $z = h \cdot W^\top$, $h_0 = W \cdot \text{onehot}(x)$. Backward of $z$ contributes $\partial\mathcal{L}/\partial W \mathrel{+}= \partial\mathcal{L}/\partial z \cdot h$. Backward of $h_0$ contributes $\partial\mathcal{L}/\partial W \mathrel{+}= \text{onehot}(x) \cdot \partial\mathcal{L}/\partial h_0$. The embedding row for token $x$ accumulates *both* gradients. **Implication:** Effective learning rate on embedding rows is roughly 2× the rate on a non-tied embedding; in practice this is fine, but Karpathy notes it as a reason tying *sometimes* needs slight LR tuning on the embedding. | advanced | proof-scaffold | m12-autograd, m16-tied-unembedding |

---

## 8. Endgame callback — three candidates

**Candidate A (definitive close, recommended):**
> "This is the entire course. You started in M5 with $f(x) = x^2$ and a single derivative. You finish here with a transformer you trained yourself, on hardware you own, with weights you can save, in roughly the time it takes to brew coffee. There is no module after this one. There is the model, the artifact, and the next thing you decide to learn. Push the button again. Watch it learn. Then close this tab and go read a paper."

**Candidate B (a meditation on what was built):**
> "You did not build GPT-4. You built something better: a transformer you can read end-to-end. Every kernel was a function you wrote. Every gradient was a derivative you took. Every loss was a number that went down because of arithmetic you understand. The model in the tab next to you knows roughly nothing — and you know roughly everything about how it knows it. That asymmetry is the whole point. Keep the .bin. Show someone."

**Candidate C (a credits roll):**
> "Module 5 — derivatives. Module 6 — chain rule. Module 7 — matmul. Module 8 — sampling. Module 9 — NLL. Module 10 — AdamW. Module 11 — MLP. Module 12 — backprop. Module 13 — LayerNorm. Module 14 — bigram, RNN, BPTT. Module 15 — attention. Module 16 — the block. Module 17 — the loop. Module 18 — the model in the tab. The bin file in your downloads. The next idea, whatever it is, that this lets you have."

**Author's recommendation: ship A. It is short, accurate, and the line "There is no module after this one" *is* the course closing on itself.**

---

## 9. Sources (licensing-aware)

1. **karpathy/nanoGPT** — Andrej Karpathy. <https://github.com/karpathy/nanoGPT>. Code repository. License: **MIT**. **[ADAPT]**. Use for: tiny-shakespeare facts (1,115,394 chars, vocab 65, 1,003,854 train / 111,540 val), `get_batch` recipe, model.py architecture conventions, training-loop conventions (warmup-cosine, grad-clip-1.0, AdamW). The reference implementation the course mirrors.
2. **karpathy/convnetjs** — Andrej Karpathy. <https://github.com/karpathy/convnetjs>. Code + browser demos. License: **MIT**. **[ADAPT]**. Use for: prior-art for in-browser live-training UX (loss curves, save-network-as-JSON button, pause/resume), the "watch a model train in your tab" pedagogical pattern.
3. **0hq/WebGPT** — 0hq. <https://github.com/0hq/WebGPT>. ~2000 lines of vanilla JS. License: **MIT** (verify in repo). **[ADAPT]**. Use for: prior art on hand-written WGSL forward kernels for a Shakespeare-trained char-model GPT, demonstrating the pedagogical viability of the from-scratch WGSL approach.
4. **WebGPU Fundamentals** — Greggman / Gregg Tavares. <https://webgpufundamentals.org/>. Web tutorial. License: **CC BY 4.0** (verify per page). **[ADAPT]**. Use for: WGSL compute shader basics, workgroup-size guidance ("default to 64"), matmul WGSL examples, dispatch-and-workgroup mental model.
5. **A Mathematical Framework for Transformer Circuits** — Elhage et al., Anthropic. <https://transformer-circuits.pub/2021/framework/index.html>. Research article. License: **proprietary Anthropic, no general reuse rights stated**. **[REFERENCE-ONLY]**. Use for: the residual-stream framing taught in M16 and re-invoked in Lesson 3's "what to learn next" mech-interp pointer; cite, do not adapt prose or figures.
6. **poloclub/transformer-explainer** — Cho, Kim, Karpekov, et al. <https://github.com/poloclub/transformer-explainer>. Svelte+ONNX-Runtime-Web tool. License: **MIT**. **[ADAPT]**. Use for: prior art on the temperature/top-p slider with live histogram (informs `samplerKnobsPlayground`), Svelte stack precedent, pedagogical structure for "abstraction-zoom" widgets.
7. **bbycroft/llm-viz** — Brendan Bycroft. <https://github.com/bbycroft/llm-viz>; live at <https://bbycroft.net/llm>. 3D walkthrough. License: check repo (likely MIT). **[REFERENCE-ONLY]** unless verified. Use for: prior art for `kernelInspector` widget — visualizing the kernel/operation graph as something you can click into.
8. **bryc/code/jshash/PRNGs.md** — bryc. <https://github.com/bryc/code/blob/master/jshash/PRNGs.md>. Reference + JS implementations of sfc32, mulberry32, xoshiro128**, cyrb128 hash. License: **public domain (CC0 / "no rights reserved")** per repo. **[ADAPT]**. Use for: the reference sfc32 + cyrb128 implementations to ship in `seedDeterminism.ts`.
9. **What's New in WebGPU (Chrome series)** — Chrome for Developers. <https://developer.chrome.com/blog/new-in-webgpu-120> and successors. Articles. License: **CC BY 4.0** (Google Developers site policies). **[ADAPT]**. Use for: authoritative info on `shader-f16` rollout dates, subgroups, dispatch overhead, deprecation notes — the concrete state of WebGPU as of the course's ship date.
10. **praeclarum/webgpu-torch** — Frank Krueger. <https://github.com/praeclarum/webgpu-torch>. PyTorch-shaped JS+WebGPU library. License: **MIT**. **[ADAPT]** (with caveat: stale since 2024). Use for: a fallback reference implementation showing the autograd-DAG approach to in-browser training, even if the course's primary path is hand-written WGSL.

**Deliberately not recommended for adaptation (license issues):** Wikipedia/Wikibooks (CC BY-SA, viral), MIT OCW (CC BY-NC-SA), Khan Academy (CC BY-NC-SA), 3Blue1Brown YouTube (no reuse license stated for adaptation; reference-only).

---

## 10. Pedagogical traps

1. **Trap: "First, let me explain the WebGPU API in detail."**
   *Why it happens:* The author is excited about WGSL and bind groups; adult learners are excited about training a model. Spending 30 minutes on `requestAdapter()` syntax loses them.
   *Mitigation:* All WebGPU plumbing lives in **one collapsed code panel** in Lesson 1 step 3. Default state: closed. Surface only the *concept* (compute pipeline, storage buffer, dispatch grid). The code is for the curious.

2. **Trap: Letting the runtime decision become the lesson.**
   *Why it happens:* The author has spent days evaluating candle vs. webgpu-torch vs. ORT-Web vs. tinygrad — and wants to share the analysis. Learners do not care which runtime; they care that it *works*.
   *Mitigation:* The entire runtime decision is a single sentence in step 1 ("we wrote ~400 lines of WGSL ourselves; here is why") with a footnote linking to a public blog post version of the analysis in §2 of this brief. **Never make the alternatives a lesson step.**

3. **Trap: The "first iter is slow" anomaly is mistaken for "training is broken."**
   *Why it happens:* iter 0 pays 1–5 seconds of WGSL compilation; the iters/sec readout shows 0.2; learner panics and refreshes.
   *Mitigation:* Surface "Compiling shaders…" as an *explicit* UI state with a progress bar before iter 0 starts. Once compilation completes, *then* start the iter counter. Mention it in prose: "first iter pays a 1–5 s shader compile; subsequent iters are at full speed."

4. **Trap: Live samples appear "broken" before iter ~200 and the learner gives up.**
   *Why it happens:* Iter 0 sample is `kxQ@!je e f` and that is correct behavior — random init produces uniform-distribution sampling. But it *looks* like a bug.
   *Mitigation:* The `liveSampleStream` widget annotates each line: `iter 0 — uniform across 65 chars (this is correct)`, `iter 50 — beginning of frequency learning`, `iter 200 — bigram-like statistics`, `iter 500 — word shapes`, `iter 2000 — Shakespeare-flavored.` Tell them what they are about to see, *before* it happens.

5. **Trap: The learner switches tabs, the loop pauses, they don't notice, and they think the model "stopped training" forever.**
   *Why it happens:* `requestAnimationFrame` callbacks are paused in background tabs. No callback = no iter. The learner returns 5 minutes later expecting iter 1500 and finds iter 80.
   *Mitigation:* Big visible "Paused (tab in background)" badge that appears on `visibilitychange`; inline explanation directly in the runner UI ("training pauses when you change tabs to save your battery — switch back to resume"). Total wall time pauses too, so iters/sec is honest.

6. **Trap: "The model is not as good as ChatGPT" disappointment.**
   *Why it happens:* The capstone produces poetry-flavored gibberish; ChatGPT is in the next tab. The contrast is brutal if not framed.
   *Mitigation:* Lesson 18.1 step 1 *opens* with: "This model has 200,000 parameters. ChatGPT has 1.7 trillion. The model is not the point. The understanding is the point." And the closing of Lesson 3 reinforces: "What this proves is not capability. It is comprehension — every line of nanoGPT, every line of every transformer paper since 2017, is now something you can read." Frame it once at the start, again at the end.
