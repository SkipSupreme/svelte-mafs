# Module 13 Research Brief — Training Dynamics & Modern Tricks

## 1. Concept dependency graph

1. **`overfitting-underfitting`** — A model overfits when training loss drops far below validation loss; it underfits when both stay high. *Prereqs: m11-neural-networks, m10-optimization.*
2. **`bias-variance`** — The expected test error decomposes into (irreducible noise) + bias² + variance, trading off against model capacity. *Prereqs: `overfitting-underfitting`.*
3. **`train-val-test-split`** — Three disjoint subsets: training fits parameters, validation tunes hyperparameters and chooses architectures, test is touched at most once for the final reported number. *Prereqs: `overfitting-underfitting`.*
4. **`generalization-gap`** — The numerical difference (train_loss − val_loss); a diagnostic, not a thing to minimize directly. *Prereqs: `train-val-test-split`.*
5. **`l2-weight-decay`** — Adds λ‖w‖² to the loss (or, in decoupled form, multiplies w by (1−λη) each step), shrinking weis: m10-optimization, m6-multivariable.*
6. **`gaussian-prior-mle`** — L2 regularization is the maximum-a-posteriori estimate under a zero-mean isotropic Gaussian prior on weights. *Prereqs: `l2-weight-decay`.*
7. **`l1-vs-l2`** — L1 (Laplace prior) produces sparse weights via corner geometry; L2 (Gaussian prior) shrinks smoothly. *Prereqs: `l2-weight-decay`, `gaussian-prior-mle`.*
8. **`dropout`** — At training time, zero each activation independently with probability p; at eval time pass through unchanged. *Prereqs: m11-neural-networks.*
9. **`inverted-dropout-scaling`** — During training, surviving activations are multiplied by 1/(1−p) so the expected pre-activation magnitude matches eval time. *Prereqs: `dropout`.*
10. **`weight-init-symmetry`** — Initializing all weights to the same value (e.g., zero) makes hidden units learn identical features; randomness is required to break symmetry. *Prereqs: m11-neural-networks, m12-backpropagation.*
11. **`variance-preservation`** — For training stability, choose Var(W) so Var(activations) and Var(gradients) stay roughly constant across depth. *Prereqs: m7-linear-algebra (matmul variance), `weight-init-symmetry`.*
12. **`xavier-glorot-init`** — For tanh/sigmoid layers, sample W with Var = 2/(n_in + n_out) (or 1/n_in for the forward-only variant). *Prereqs: `variance-preservation`.*
13. **`he-kaiming-init`** — For ReLU layers, sample W with Var = 2/n_in to compensate for ReLU killing half the signal. *Prereqs: `variance-preservation`, `xavier-glorot-init`.*
14. **`batch-norm`** — Normalize each pre-activation across the batch dimension, then learn affine γ,β; uses batch statistics in training and stored running means in eval. *Prereqs: m12-backpropagation, `variance-preservation`.*
15. **`bn-train-vs-eval`** — Training uses minibatch μ_B, σ²_B; eval uses an EMA accumulated during training. Forgetting `model.eval()` is a common bug. *Prereqs: `batch-norm`.*
16. **`bn-as-landscape-smoother`** — BatchNorm helps less by removing "internal coving the loss landscape and gradients more Lipschitz/predictable (Santurkar 2018). *Prereqs: `batch-norm`.*
17. **`layer-norm`** — Normalize across the feature dimension *per token*, independent of batch; preferred for variable-length sequences and small batches. *Prereqs: `batch-norm`.*
18. **`bn-vs-ln-when`** — BN is the default for vision/CNNs with large stable batches; LN is the default for transformers because batch statistics are unstable across padded/variable-length sequences and at inference time. *Prereqs: `batch-norm`, `layer-norm`.*
19. **`residual-connection`** — `y = x + F(x)`: an identity shortcut around a sub-block so the block only learns the *correction*. *Prereqs: m12-backpropagation, m11-neural-networks.*
20. **`gradient-flow-through-residuals`** — Backprop through a residual gives `dL/dx = dL/dy · (I + F'(x))`, so gradients can never vanish through the identity path. *Prereqs: `residual-connection`, m12-backpropagation.*
21. **`pre-ln-vs-post-ln`** — Pre-LN places normalizatiothe residual branch (`x + F(LN(x))`); post-LN normalizes the sum (`LN(x + F(x))`). Pre-LN is what GPT-2/3 and modern LLMs use because it trains stably without warmup tricks. *Prereqs: `residual-connection`, `layer-norm`.*
22. **`residual-init-scaling`** — In a deep residual stack, initialize the residual *output* projections with std scaled by 1/√(N_layers) (or 1/√(2·N_layers)) to keep variance of the residual stream constant with depth. *Prereqs: `residual-connection`, `he-kaiming-init`.*
23. **`lr-warmup`** — Ramp the learning rate linearly from ~0 to its peak over the first few hundred-thousand steps, then decay. Necessary for Adam-trained transformers because early gradients have high variance and skewed second-moment estimates. *Prereqs: m10-optimization (Adam, schedules).*
24. **`gradient-clipping`** — When ‖g‖ > τ, rescale g ← g · τ/‖g‖. A pragmatic ceiling, not a fix for bad architecture. *Prereqs: m12-backpropagation, m7-linear-algebra (norms).*
25. **`loss-curve-pathologiesary: clean fit, overfitting fork, plateau, divergence spike, "dead" curve (no learning), validation noise, double descent. *Prereqs: `train-val-test-split`, `generalization-gap`.*

---

## 2. Canonical worked examples

### 2.1 Variance through one linear layer (the Glorot derivation in 90 seconds)
**Problem.** A layer computes $z = \sum_{i=1}^{n} w_i x_i$ where the $x_i$ are i.i.d. with mean 0, variance $\sigma_x^2$, and the $w_i$ are i.i.d. with mean 0, variance $\sigma_w^2$, all independent. What does $\text{Var}(z)$ equal? What variance of $w$ keeps $\text{Var}(z) = \sigma_x^2$?

**Solution.** $\text{Var}(z) = n \cdot \text{Var}(w_i x_i) = n \cdot E[w_i^2]E[x_i^2] = n \sigma_w^2 \sigma_x^2$. To preserve variance, pick $\sigma_w^2 = 1/n$. For ReLU, half the inputs are zeroed in expectation, so we double it: $\sigma_w^2 = 2/n$ (He init).

**Pedagogical point.** Initialization is not folklore — it falls out of one line of variance algebra.

**Common mistake.** Forgetting that independence is required to dp the cross terms; or using $E[w]=0$ to claim $E[w^2]=0$ (it equals the variance, not zero).

### 2.2 Activation collapse at depth 10 (Karpathy makemore Part 3)
**Problem.** Build a 10-layer MLP with tanh activations, 200 hidden units per layer, weights initialized as $\mathcal{N}(0, 1)$. Pass a batch of unit-variance inputs through. By layer 10, what is the empirical distribution of activations, and what fraction of tanh derivatives is below 0.01? Now rescale weights to $\mathcal{N}(0, 5/3 \cdot \sqrt{1/n})$ and repeat.

**Solution.** With $\mathcal{N}(0,1)$ weights, the pre-activation variance multiplies by ~200 per layer; tanh saturates to ±1 by layer 2–3, and almost every neuron has derivative essentially 0 — the network is dead. With Kaiming-tanh scaling ($\text{gain}=5/3$, $\sigma_w^2 = (5/3)^2 / n_{\text{in}}$), activation variance stays roughly constant (≈1) at every depth and gradients flow.

**Pedagogical point.** Bad init *silently* breaks training — there is no error message, the loss jes to drop.

**Common mistake.** Believing Adam will "find" the right scale — it can't when 99% of derivatives are numerically zero from step 0.

### 2.3 Inverted dropout: the expected-value contract
**Problem.** A neuron outputs $a = 4$. Dropout with $p = 0.25$ (probability of dropping). Show that without scaling, train and eval expectations differ. Derive the scale factor that makes them match.

**Solution.** During training, $E[\tilde a] = (1-p) \cdot a = 0.75 \cdot 4 = 3$. During eval (no dropout), the value is 4. Mismatch. Inverted dropout multiplies survivors by $1/(1-p) = 4/3$ during training, so $E[\tilde a] = (1-p) \cdot a/(1-p) = 4$. Eval is then unchanged.

**Pedagogical point.** Dropout is not "make the network smaller" — it is "make the training-time and eval-time distributions match in expectation while injecting noise."

**Common mistake.** Multiplying by $(1-p)$ at eval time *and* using a framework that already does inverted dropout — silently halving the activation magnitudes.

### 2.hNorm with batch size 1 at eval
**Problem.** A network is trained with BN and `momentum=0.1` on running stats. Inference is then run on a single example with `model.eval()`. Then the same example is run with `model.train()`. Why do you get a sensible answer in eval mode and garbage in train mode?

**Solution.** In eval mode, BN uses the *stored* running mean and variance accumulated across the entire training set, applied deterministically to the single example. In train mode, BN computes batch statistics from the current batch — which here has size 1, so $\mu_B = x$ exactly and $\sigma_B^2 = 0$, meaning $\hat x_i = (x_i - x_i)/\sqrt{\epsilon} = 0$. Every output is just β. The whole feature dimension is zeroed.

**Pedagogical point.** BN's behavior at train vs. eval is not symmetric, and this asymmetry is the most common BN bug.

**Common mistake.** Believing `model.eval()` is cosmetic; not realizing dropout and BN both gate on `self.training`.

### 2.5 L2 as MAP under a Gaussian prior
**Problem.** Show t minimizing $-\log p(\mathcal{D}|w) + \frac{\lambda}{2}\|w\|_2^2$ is equivalent to MAP estimation $\arg\max_w p(w|\mathcal{D})$ under prior $w \sim \mathcal{N}(0, \sigma^2 I)$, and derive the relationship between $\lambda$ and $\sigma^2$.

**Solution.** $p(w|\mathcal{D}) \propto p(\mathcal{D}|w) p(w)$. Take negative log: $-\log p(w|\mathcal{D}) = -\log p(\mathcal{D}|w) - \log p(w) + C$. With Gaussian prior, $-\log p(w) = \frac{1}{2\sigma^2}\|w\|^2 + C'$. Matching gives $\lambda = 1/\sigma^2$. A *narrower* prior (smaller $\sigma^2$) means *larger* $\lambda$, i.e., stronger shrinkage.

**Pedagogical point.** Regularization isn't a hack bolted onto the loss; it is the prior the modeler is asserting about the world.

**Common mistake.** Thinking $\lambda$ and prior variance are proportional, when they are *inverse* — small prior variance = strong regularization.

### 2.6 The residual-connection gradient (the ResNet identity in one line)
**Problem.** A residual block computes $y = x + F(x; \theta)$. Compute $artial L / \partial x$ in terms of $\partial L / \partial y$. What does this say about gradient flow at initialization, when $F$ is a freshly-initialized small perturbation?

**Solution.** $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y}\left(I + \frac{\partial F}{\partial x}\right)$. At init, if $F$ is small (e.g., $F$'s output projection is initialized near zero), $\partial F / \partial x \approx 0$, so $\partial L / \partial x \approx \partial L / \partial y$. Gradients pass through *unattenuated*, regardless of depth.

**Pedagogical point.** Residuals don't just let you "stack more layers." They give every layer a privileged identity-like gradient path that backprop cannot break.

**Common mistake.** Thinking residuals work because "the network can ignore layers" — they work because gradients see an identity Jacobian on the shortcut.

### 2.7 LayerNorm of a token vector
**Problem.** A token has feature vector $x = [1, 2, 3, 4, 5, 6]$. Compute $\hat x_i = (x_i - \mu)/\sqrt{\sigma^2 + \epson}$ with $\epsilon = 0$. Confirm that mean of $\hat x$ is 0 and variance is 1.

**Solution.** $\mu = 3.5$. $\sigma^2 = \frac{1}{6}\sum(x_i - 3.5)^2 = (6.25 + 2.25 + 0.25 + 0.25 + 2.25 + 6.25)/6 = 17.5/6 \approx 2.917$, so $\sigma \approx 1.708$. $\hat x \approx [-1.464, -0.878, -0.293, 0.293, 0.878, 1.464]$. Mean is 0, variance is 1 by construction.

**Pedagogical point.** LayerNorm acts on a single example — it does not care about the batch, which is exactly why transformers prefer it.

**Common mistake.** Computing the variance using the unbiased $(n-1)$ denominator instead of $n$; PyTorch's LayerNorm uses biased variance.

### 2.8 BatchNorm as a special case of "what gets normalized"
**Problem.** Given a tensor of shape (B, C, H, W) — a batch of feature maps — over which axes does BatchNorm reduce, and over which does LayerNorm (in the per-token sense) reduce? Predict what happens when batch size drops from 64 to 2.

**Solution.** BN reduces over (B, H, W) for each channel C, computing C means andiances from B·H·W samples. LayerNorm in vision is typically over (C, H, W) per example. With batch size 2, BN's per-channel statistics are estimated from only $2 \cdot H \cdot W$ samples, which is noisy; the regularization "noise" becomes signal-destroying. Recommendation: use GroupNorm or LayerNorm.

**Pedagogical point.** "Which axes do you reduce over?" is the entire taxonomy of normalization layers.

**Common mistake.** Memorizing the names instead of the shape arithmetic.

---

## 3. Common misconceptions

1. **"Dropout makes the network smaller."** Natural because the picture shows half the nodes crossed out. Kill it: at eval time *all* neurons fire; the inverted-1/(1-p) scaling exists precisely to keep the *full* network's expected pre-activation magnitude equal to what it saw during training. Dropout is noise injection, not architectural surgery.

2. **"BatchNorm just centers the data."** Plausible because the formula starts with subtracting the mean. Reframe: it normalizes *and* learns γ, β, *amaintains EMA running stats, *and* (per Santurkar 2018) its real benefit is making the loss landscape smoother / gradients more predictive, not removing covariate shift.

3. **"More parameters = more overfitting."** True for shallow models but spectacularly wrong in the deep-learning regime. Counterexample: deep double descent — past the interpolation threshold, test error *drops again* as parameters grow. Show OpenAI's deep-double-descent curves.

4. **"L2 regularization is just L1 squared."** They sound similar but the geometry is different: L2's prior is Gaussian (smooth), L1's is Laplace (corner at 0). The corner is what drives sparsity. Visualize the unit balls: a circle versus a diamond; gradient descent meets the diamond at corners.

5. **"The validation set is for tuning, the test set is just another validation set."** This is the leakage that voids the test claim. Reframe: every time you peek at test loss, it becomes part of your training signal. The test set is a single irreversible bullet you fe at the end. Treat it like a sealed envelope.

6. **"Weight init doesn't matter, the optimizer fixes it."** Counterexample (worked in §2.2): a 10-layer tanh net with $\mathcal{N}(0,1)$ weights has saturated activations and zero gradients from step 0. Adam cannot move parameters whose gradient is exactly 0. Demo this live.

7. **"Residuals are about going deeper."** They are about *gradient flow*. Show the $\partial L/\partial x = \partial L/\partial y \cdot (I + \partial F/\partial x)$ identity: the `I` term means gradient never has to multiply through F to reach earlier layers.

8. **"Warmup is just a heuristic / cargo cult."** Reframe: Adam's second-moment estimate $\hat v_t$ has very high variance for small $t$, making per-parameter step sizes unreliable. Plus, in deep networks at init the gradients themselves have skewed distributions. Warmup is a principled response: take small steps until your statistics stabilize.

9. **"BatchNorm and LayerNorm normalize the same thing in different orders."** No �hey reduce over different axes. BN: across the batch, per feature. LN: across features, per example. The choice has nothing to do with order; it has to do with *which dimension's statistics are stable*.

10. **"Pre-LN vs post-LN is a stylistic choice."** No — original Transformer used post-LN and required learning-rate warmup to train at all; pre-LN trains stably without warmup at much greater depth and is what every modern LLM (GPT-2/3, LLaMA family, Chinchilla) uses. Recommend pre-LN unconditionally for the course.

11. **"Gradient clipping fixes exploding gradients."** It clips the *symptom*. Exploding gradients are usually caused by bad init, missing normalization, or recurrent dynamics with eigenvalues > 1. Clipping is a safety belt; the seatbelt does not fix bad driving.

12. **"Regularization means L2."** Regularization is *anything that constrains the hypothesis space or biases search toward simpler functions*: weight decay, dropout, BN's batch noise, early stopping, data augmentation, even SGD's ise. Don't conflate the umbrella with one item under it.

---

## 4. Interactive widget suggestions

### 4.1 `varianceFanOut`
- **Manipulate.** Drag a "weight std" knob attached to a stack of 12 layers; alternatively, click any single layer to scrub *its* std independently.
- **Live updates.** Histogram of activations at every layer (12 mini-histograms in a column) plus a "saturation %" readout per layer, plus the average absolute gradient at layer 1 (the deepest-from-output) shown as a number that flashes red when below 1e-7.
- **Concept it makes tangible.** Variance preservation. The learner *sees* activations explode or collapse as they drag the std away from $\sqrt{2/n}$; the gradient at layer 1 visibly dies.
- **Why a slider+number is insufficient.** A single number can't show the layer-by-layer cascade. The visual signature of "saturated tanh wall" or "exploding ReLU" is what builds intuition; a scalar can't communicate it.
- **Prior art.** deeplearning.ai's "Initializing neural networks" article (https://www.deeplearning.ai/ai-notes/initialization/), Karpathy's makemore Part 3 activation-histogram column, TensorFlow Playground (https://playground.tensorflow.org/) for general feel.

### 4.2 `dropoutMaskScrub`
- **Manipulate.** A small MLP rendered as nodes; the user scrubs the dropout rate p from 0 to 0.9, *and* clicks an "advance step" button to resample the mask. A toggle switches between "show training mode" (mask + ×1/(1−p) scaling) and "show eval mode" (no mask, no scaling).
- **Live updates.** The dropped neurons gray out; surviving neurons display their scaled value next to them; a small bar at the top tracks E[output] across 50 sampled masks and confirms it equals the eval-mode output to within ε.
- **Concept it makes tangible.** Inverted-dropout's expected-value contract — that the scaling exists *to* make train and eval match.
- **Why a slider+number is insufficient.** You need to see the random masks resampling and the running-mean readout converge to the eval value. The "wow" moment ising the average bar match the eval bar exactly.
- **Prior art.** No great existing one — deeplearning.ai's regularization article shows dropout statically. This widget would be original; loosely inspired by Nicky Case's "To Build a Better Ballot" use of repeated sampling.

### 4.3 `bnTrainEvalToggle`
- **Manipulate.** A 1D feature scatterplot for one channel, with batch size as a draggable handle (1 to 64). A toggle for `train` vs `eval` mode. A "training step" button that updates the running EMA.
- **Live updates.** Running mean/var bars (gray, slowly tracking), batch mean/var bars (orange, jumping each click), and the *output* of BN displayed for one fixed test point under each mode. As batch size drops to 1, the train-mode output collapses to β; eval-mode output stays sane.
- **Concept it makes tangible.** Why BN with batch size 1 in train mode produces zeros, why eval mode is robust, why the running stats *lag* during training and can be wrong if you haven't trained long enough.
- **Why a slider+numbis insufficient.** You need to *see* batch stats jitter while running stats stabilize, and you need the toggle to watch the same point produce different outputs in two modes.
- **Prior art.** None really good. The PyTorch docs explain it in prose. Distill's Activation Atlas has the visual style we want.

### 4.4 `residualGradientFlow`
- **Manipulate.** A vertical stack of 30 blocks. Toggle "skip connections on/off" globally, or per-block. Change the activation function (tanh, ReLU). Run backprop on a fake loss.
- **Live updates.** Heat-bar showing the magnitude of $\partial L / \partial a^{(k)}$ at every depth k, in log scale. With residuals on, the bar stays roughly uniform; with residuals off and tanh/30 layers deep, the bar collapses to near-zero at the input side.
- **Concept it makes tangible.** Residuals as gradient highways — the identity Jacobian guaranteeing flow.
- **Why a slider+number is insufficient.** The gradient-magnitude heat-column is the *whole point*; a single norm doesn't show where t signal dies.
- **Prior art.** ResNet paper figure 1 shows the train-error degradation but is static. The Hugging Face computer-vision course has a static schematic. This widget makes the abstract "vanishing gradient" claim *operational*.

### 4.5 `lossCurveDoctor`
- **Manipulate.** A dropdown of 8 prebuilt scenarios ("learning rate too high," "missing dropout, small dataset," "BN in eval mode during training," "deep net without residuals," "warmup omitted, Adam"). The learner clicks a scenario and is shown a train/val loss curve pair *without a label*. They drag from a palette of "diagnosis cards" (overfitting, divergence, plateau, dead-init, etc.) onto the curve.
- **Live updates.** A green checkmark or red X immediately; on success, an annotated overlay highlights the curve features that justify the diagnosis (e.g., "val curve forks upward at epoch 30 → overfitting begins here").
- **Concept it makes tangible.** Reading pathologies. Adult learners need to *see* the difference between overfitting and dirgence and a plateau.
- **Why a slider+number is insufficient.** This is a perception/diagnosis skill, not a parameter sweep. Learners get good at it through pattern matching with feedback.
- **Prior art.** Karpathy's "Recipe for Training Neural Networks" lists pathologies in prose. Weights & Biases reports show curves but not as a guessing game. This widget is the diagnostic gym fast.ai students wish they'd had.

### 4.6 `priorPenaltyEquivalence`
- **Manipulate.** A 2D weight space with a true minimum at a noisy spot. Drag the regularization strength λ; toggle between "L2 penalty" view and "Gaussian prior" view; toggle between "L1 penalty" view and "Laplace prior" view.
- **Live updates.** Loss-surface contours, prior-density contours (concentric circles for Gaussian, concentric diamonds for Laplace), the MAP solution dot, and a side panel that shows σ² = 1/λ updating in real time. Switching between L1 and L2 visibly changes the corner geometry and slides the solution onto an axis (or off it).
- **Concit makes tangible.** Regularization *is* a Bayesian prior; the prior's variance and the penalty's λ are reciprocals.
- **Why a slider+number is insufficient.** The geometric meaning ("L1 corner pulls solutions onto axes") only lands visually.
- **Prior art.** mlu-explain's bias-variance article has similar contour visuals (CC-BY-SA → reference-only). 3B1B's series on Bayes uses prior-density overlays.

### 4.7 `warmupStabilizer`
- **Manipulate.** A pre-LN vs post-LN toggle; a warmup-steps slider (0, 100, 500, 2000); peak learning rate slider; optimizer (Adam vs SGD) toggle.
- **Live updates.** Live-running training loss for a tiny 6-layer transformer-ish net on a toy task, plus a histogram of Adam's $\hat v_t$ values across parameters during the first 500 steps. With Post-LN + Adam + no warmup → loss diverges; with Post-LN + Adam + 500 warmup steps → trains. With Pre-LN + Adam + no warmup → trains. The $\hat v_t$ histogram is visibly degenerate (heavy-tailed) early.
- **Concept it makes tangible.**up exists, *and* why pre-LN reduces (but doesn't eliminate) the need for it.
- **Why a slider+number is insufficient.** The interaction between three architectural choices (Adam, post-LN, no-warmup → divergence) is irreducibly multi-dimensional.
- **Prior art.** RBC Borealis's "Transformers III" tutorial has the static figure (https://rbcborealis.com/research-blogs/tutorial-17-transformers-iii-training/); we make it interactive.

---

## 5. Key formulas

### Regularization
- `\mathcal{L}_{\text{total}}(w) = \mathcal{L}_{\text{data}}(w) + \frac{\lambda}{2}\|w\|_2^2`
- `w_{t+1} = (1 - \eta\lambda) w_t - \eta \nabla \mathcal{L}_{\text{data}}(w_t)`
- `w_{\text{MAP}} = \arg\min_w \left[ -\log p(\mathcal{D}\mid w) + \frac{1}{2\sigma^2}\|w\|_2^2 \right], \quad \lambda = 1/\sigma^2`
- `p(w) = \mathcal{N}(w; 0, \sigma^2 I) \;\Leftrightarrow\; \text{L2}; \qquad p(w) = \text{Laplace}(0, b) \;\Leftrightarrow\; \text{L1}`
- `\mathcal{L}_{\text{L1}}(w) = \mathcal{L}_{\text{data}}(w) + \lambda \sum_i |w_i|`

### Dropout `r_i \sim \text{Bernoulli}(1-p)`
- `\tilde a_i = \frac{r_i}{1-p}\, a_i \quad \text{(training, inverted dropout)}`
- `\tilde a_i = a_i \quad \text{(inference)}`
- `\mathbb{E}[\tilde a_i] = a_i`

### Initialization (variance preservation)
- `\text{Var}(z) = n_{\text{in}} \cdot \text{Var}(W) \cdot \text{Var}(x)`
- `\text{Xavier/Glorot:}\; \text{Var}(W) = \frac{2}{n_{\text{in}} + n_{\text{out}}}`
- `\text{Xavier (forward only):}\; \text{Var}(W) = \frac{1}{n_{\text{in}}}`
- `\text{He/Kaiming (ReLU):}\; \text{Var}(W) = \frac{2}{n_{\text{in}}}`
- `\text{Tanh gain:}\; \sigma_W = \frac{5/3}{\sqrt{n_{\text{in}}}}`
- `\text{Residual scaling (GPT-2):}\; \sigma_W = \frac{0.02}{\sqrt{2 N_{\text{layers}}}}\;\text{on residual output projections}`

### Batch Normalization
- `\mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i`
- `\sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m} (x_i - \mu_B)^2`
- `\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}`
- `y_i = \gamma \hat{x}_i + \beta`
- `\mu_{\text{run}} \leftarrow (1-\alpha)\,\mu_{\text{run}} + \alpha\, \mu_B`
- `\sigma^2_{\text{run}} \leftarrow (1-\alpha)\,\sigma^2_{\text{run}} + \alpha\, \sigma_B^2`
- `\hat{x}_{\text{eval}} = \frac{x - \mu_{\text{run}}}{\sqrt{\sigma^2_{\text{run}} + \epsilon}}`

### Layer Normalization
- `\mu = \frac{1}{H}\sum_{j=1}^{H} x_j`
- `\sigma^2 = \frac{1}{H}\sum_{j=1}^{H} (x_j - \mu)^2`
- `\hat{x}_j = \frac{x_j - \mu}{\sqrt{\sigma^2 + \epsilon}}`
- `y_j = \gamma_j\, \hat{x}_j + \beta_j`
- `\text{RMSNorm:}\; y_j = \frac{x_j}{\sqrt{\frac{1}{H}\sum_k x_k^2 + \epsilon}}\,\gamma_j`

### Residual / skip connections
- `y = x + F(x; \theta)`
- `\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y}\left(I + \frac{\partial F(x)}{\partial x}\right)`
- `\text{Pre-LN block:}\; y = x + F(\text{LN}(x))`
- `\text{Post-LN block:}\; y = \text{LN}(x + F(x))`

### Warmup / schedule
- `\eta_t = \eta_{\max}\cdot \min\!\left(\frac{t}{t_{\text{warmup}}},\; 1\right)\quad\text{(linear warmup)}`
- `\eta_t = \frac{1}{2}\eta_{\max}\bigl(1 + \cos\bigl(\pi\,\tfrac{t - t_{\text{warmup}}}{T - t_{\text{warmup}}}\bigr)\bigr) \quad \text{(cosine decay)}`
- `\text{Original Transformer schedule:}\;\eta_t = d_{\text{model}}^{-1/2}\cdot \min\!\bigl(t^{-1/2},\, t\cdot t_{\text{warmup}}^{-3/2}\bigr)`

### Gradient clipping
- `g \leftarrow g \cdot \min\!\left(1,\, \frac{\tau}{\|g\|_2}\right)`

---

## 6. Lesson decomposition

The module splits into **5 lessons**, ~15 minutes each.

### Lesson 13.1 — "When your model memorizes the training set"
**Summary.** Train and validation curves are diagnostic instruments; learn to read them.

**Steps (10):**
1. *Hook: a 5-parameter and a 5000-parameter polynomial fitting noisy points.* Prose+widget; learner watches both fit the same data, but only one *generalizes*.
2. *The bias–variance decomposition.* Prose, derives $E[(y-\hat f)^2] = \sigma^2 + \text{bias}^2 + \text{variance}$.
3. *Define overfitting and underfitting in one sentence each.* Prose.
4. **StepCheck:** "If train loss = 0.05 and val loss = 0.50, what is the generalization gap?" → **0.45**.rain, val, test: three jobs, three sets.* Prose.
6. *The cardinal sin: peeking at test.* Prose, with a concrete leakage scenario.
7. **StepCheck:** "You have 10,000 examples and want a 70/15/15 split. How many test examples?" → **1500**.
8. *Five canonical loss-curve shapes.* Widget: `lossCurveDoctor`, scenarios 1–5.
9. *Diagnosis practice.* Widget: `lossCurveDoctor`, scenarios 6–8.
10. *Where this shows up in the transformer.* Callback: training a tiny char-level transformer, you'll watch these exact curves; recognizing them in real time is the diagnostic skill.

**Widgets.** `lossCurveDoctor`. **Estimated minutes:** 12.

---

### Lesson 13.2 — "Two ways to keep weights small (and what they really mean)"
**Summary.** L2, L1, and dropout, framed as priors and as expected-value contracts.

**Steps (12):**
1. *The naive fix: cap weight magnitudes.* Prose.
2. *Weight decay = L2.* The training rule $w \leftarrow (1-\eta\lambda)w - \eta\nabla L_{\text{data}}$.
3. *L2 has a Bayesian story.* Prose: Gaussia MAP, the σ²=1/λ identity.
4. *Geometric picture: L2 vs L1.* Widget: `priorPenaltyEquivalence`.
5. **StepCheck:** "If you believe weights are tightly clustered around 0 (small σ²), is λ large or small?" → **large**.
6. *Decoupled weight decay (AdamW).* Prose, brief.
7. *Dropout: the radical idea.* Prose.
8. *Why scaling matters.* Worked example §2.3.
9. *Inverted dropout in code (3 lines, PyTorch-style).* Code block.
10. **StepCheck:** "p = 0.2 dropout. A surviving activation was 5.0. What is its scaled value during training?" → **6.25**.
11. *Watch the expectation match.* Widget: `dropoutMaskScrub`.
12. *Where this shows up in the transformer.* The transformer block applies dropout to the attention output and to the MLP output; weight decay applies to all projection matrices. We will use AdamW with $\lambda = 0.1$.

**Widgets.** `priorPenaltyEquivalence`, `dropoutMaskScrub`. **Estimated minutes:** 14.

---

### Lesson 13.3 — "Why your gradients explode"
**Summary.** Initialization is the diffe training and not training.

**Steps (12):**
1. *Hook.* A 10-layer tanh net with $\mathcal{N}(0,1)$ weights. Loss won't move. Why?
2. *Variance through one linear layer.* Worked example §2.1.
3. *Stack ten of them.* Widget: `varianceFanOut`, weights = $\mathcal{N}(0,1)$.
4. *The variance-preservation rule.* Pick Var(W) = 1/n_in.
5. **StepCheck:** "Layer has 256 inputs, tanh activations. Using Xavier (forward-only), what is Var(W)?" → **1/256 ≈ 0.0039**.
6. *Why Xavier breaks for ReLU.* Half the inputs are zero; you've lost half the variance.
7. *He/Kaiming: 2/n_in.* Prose + formula.
8. **StepCheck:** "Layer has 512 inputs, ReLU. He init: what is Var(W)?" → **2/512 = 0.00390625**.
9. *Symmetry breaking.* Why all-zeros fails: every neuron sees identical gradients, learns identical features.
10. *Activation distributions at depth 10, fixed.* Widget: `varianceFanOut`, weights = He.
11. *In modern transformers.* Init residual output projections with std = 0.02/√(2·N_layers); see GPT-2 paper, nanoGPT mo. *Where this shows up in the transformer.* Our transformer's `c_proj.weight` matrices use exactly the GPT-2 scaled init formula.

**Widgets.** `varianceFanOut`. **Estimated minutes:** 15.

---

### Lesson 13.4 — "Normalize what, exactly?"
**Summary.** BatchNorm, LayerNorm, and the axis you reduce over.

**Steps (13):**
1. *The motivation.* In a deep net, even with good init, activations drift during training. Show makemore-style activation distribution evolution.
2. *BatchNorm: the formula.* $\hat x = (x-\mu_B)/\sqrt{\sigma_B^2 + \epsilon}$, then $y = \gamma \hat x + \beta$.
3. *Why γ and β?* So the network can *undo* the normalization if useful.
4. *Train vs eval — the most common bug.* Prose; introduce running stats.
5. **StepCheck:** "In eval mode, BN normalizes using…?" → **the running mean and variance accumulated during training** (free-text or multiple-choice with one correct answer).
6. *Batch size 1 at train time.* Worked example §2.4.
7. *Toggle the modes.* Widget: `bnTrainEvalToggle`.N is really doing (Santurkar 2018).* Prose: it's not internal covariate shift; it's a smoother loss landscape.
9. *LayerNorm: same formula, different axis.* Worked example §2.7.
10. **StepCheck:** "Token vector x = [1, 2, 3]. After LN with γ=1, β=0, ε=0, what is $\hat x_2$?" → **0** (the middle of [-1.22, 0, 1.22] up to rounding; since μ=2, $\hat x_2 = (2-2)/\sigma = 0$).
11. *Why transformers use LN, not BN.* Variable sequence lengths, padded tokens corrupt batch statistics, single-example inference. Recommend LN unconditionally for sequence models.
12. *RMSNorm — the simplification.* LLaMA et al. drop the mean term and just rescale by RMS.
13. *Where this shows up in the transformer.* Pre-LN architecture: every sub-block is `x + F(LN(x))`. Two LayerNorms per transformer block. We use **pre-LN**.

**Widgets.** `bnTrainEvalToggle`. **Estimated minutes:** 15.

---

### Lesson 13.5 — "The tricks that make depth possible"
**Summary.** Residuals, warmup, gradient clipping — and why they're load-bearansformer.

**Steps (13):**
1. *The degradation problem.* Plain 56-layer net is *worse* than a 20-layer net. Show the ResNet-paper figure (description, not embedded).
2. *Residuals: y = x + F(x).* Prose.
3. *The gradient identity.* Worked example §2.6.
4. *See gradient flow with and without residuals.* Widget: `residualGradientFlow`.
5. **StepCheck:** "In a residual block at initialization where F's last layer is initialized near zero, what is $\partial L/\partial x$ approximately equal to?" → **$\partial L/\partial y$**.
6. *Pre-LN vs post-LN.* Prose. Recommend pre-LN. Connection: pre-LN keeps the residual stream's variance bounded; post-LN does not.
7. *Residual init scaling.* The 1/√(2·N_layers) trick for keeping the residual stream's variance from growing with depth.
8. *Adam's early instability.* Prose: $\hat v_t$ has high variance for small t.
9. *Warmup — what it does.* Linear ramp from 0 → η_max over t_warmup steps.
10. *Toggle warmup on/off, with and without pre-LN.* Widget: `warmupStab. **StepCheck:** "We use t_warmup = 200 steps and peak η = 6e-4. At step 50, the learning rate is?" → **1.5e-4**.
12. *Gradient clipping — the seatbelt.* When ‖g‖ > τ, rescale. Prose: this is a band-aid, not a cure.
13. *Where this shows up in the transformer.* Our transformer block uses pre-LN, residual connections around both attention and MLP, residual-output init scaling 0.02/√(2·N_layers), AdamW with linear warmup over 200 steps, and gradient clipping at norm 1.0.

**Widgets.** `residualGradientFlow`, `warmupStabilizer`. **Estimated minutes:** 16.

---

## 7. Problem bank

1. **(novice / computation / `variance-preservation`, `xavier-glorot-init`)** A linear layer has 1024 inputs and tanh activation. Using Xavier (forward-only) initialization, what is $\sigma_W$? **Answer:** $\sqrt{1/1024} \approx 0.03125$.

2. **(novice / computation / `he-kaiming-init`)** A linear layer has 256 inputs and ReLU activation. Using He init, what is $\text{Var}(W)$? **Answer:** $2/256 = 0.0078125$.

3. **(novtion / `inverted-dropout-scaling`)** Dropout with $p = 0.3$. A surviving activation was 7.0. What is its scaled value during training? **Answer:** $7.0 / (1 - 0.3) = 10.0$.

4. **(novice / computation / `lr-warmup`)** Linear warmup over 1000 steps to peak η = 3e-4. What is the learning rate at step 250? **Answer:** $0.25 \times 3 \times 10^{-4} = 7.5 \times 10^{-5}$.

5. **(novice / computation / `layer-norm`)** Apply LayerNorm to $x = [2, 4, 6, 8]$ with $\gamma = 1, \beta = 0, \epsilon = 0$. What is $\hat x_4$? **Answer:** $\mu = 5$, $\sigma^2 = 5$, $\sigma \approx 2.236$, $\hat x_4 = (8-5)/2.236 \approx 1.342$.

6. **(novice / computation / `l2-weight-decay`, `gaussian-prior-mle`)** L2 regularization corresponds to a Gaussian prior with σ² = 4. What is λ? **Answer:** $\lambda = 1/\sigma^2 = 0.25$.

7. **(intermediate / interpretation / `loss-curve-pathologies`)** Training loss drops smoothly to 0.1; validation loss drops to 0.4 at epoch 20, then begins to *rise*, reaching 0.7 by epoch 50. What is happg, and what would you do? **Answer:** Classic overfitting. Apply early stopping at the validation minimum (~epoch 20), or add regularization (dropout, weight decay), or get more data.

8. **(intermediate / debugging / `bn-train-vs-eval`)** Your model trains fine but produces nonsense at inference. You discover you forgot `model.eval()`. Concretely, what computation does BN do during inference *with* `model.eval()`, vs. *without* it (in train mode), on a single test example? **Answer:** With eval mode, BN normalizes by stored running mean/variance. Without (still in train mode), BN computes batch statistics from the single example: $\mu_B = x$, $\sigma_B^2 = 0$, so $\hat x = 0$ for every channel and the output is just $\beta$.

9. **(intermediate / interpretation / `weight-init-symmetry`)** A coworker initializes all weights to 0.01 (a small constant, not random). They report training fails — every neuron in a layer ends up identical. Explain in two sentences. **Answer:** Identical weights produce identicaactivations; identical activations produce identical gradients; gradient descent then updates them identically. The hidden units never differentiate, so the layer collapses to a single effective unit.

10. **(intermediate / construction / `xavier-glorot-init`, `variance-preservation`)** Show that for a linear layer $z = \sum_{i=1}^n w_i x_i$ with i.i.d. zero-mean weights and inputs, $\text{Var}(z) = n \cdot \text{Var}(W) \cdot \text{Var}(x)$. State which independence assumption you used and where. **Answer:** $\text{Var}(z) = \sum \text{Var}(w_i x_i) = \sum E[w_i^2 x_i^2] - 0 = n \cdot E[w^2]E[x^2] = n \cdot \text{Var}(W)\text{Var}(x)$. Independence between $w_i$ and $x_i$ used to factor $E[w_i^2 x_i^2]$; mutual independence of the n terms used to drop cross-covariances in the sum's variance.

11. **(intermediate / debugging / `batch-norm`)** A vision model trains well at batch size 64 but degrades sharply at batch size 4. Train and val curves both look noisier. What's the most likely culprit and what's the fix? **Answer:** BatchNorm with small batches has noisy per-batch statistics that act as overpowered regularization noise and produce poor running estimates. Fix: switch to GroupNorm or LayerNorm, or use SyncBN across devices.

12. **(intermediate / computation / `variance-preservation`, deep)** A 10-layer linear net (no nonlinearity, fan-in 100 each layer) is initialized with $\text{Var}(W) = 1/50$ everywhere. Inputs have unit variance. What is the variance of the layer-10 output? **Answer:** Each layer multiplies variance by $n \cdot \text{Var}(W) = 100/50 = 2$. After 10 layers, variance is $2^{10} = 1024$. Activations explode.

13. **(intermediate / interpretation / `gradient-flow-through-residuals`)** Why does adding a single residual connection in front of a deep stack fix the vanishing-gradient problem, even if the stack has saturating activations? **Answer:** $\partial L/\partial x = \partial L/\partial y \cdot (I + J_F)$. The $I$ term is a gradient highway: even if every layer's local Jacobian is small (saturating), the identity ensures gradient magnitude survives; the residual block can only *add to* gradient flow, never multiplicatively shrink it.

14. **(advanced / debugging / `lr-warmup`, `pre-ln-vs-post-ln`)** A graduate student is training a 12-layer post-LN transformer with Adam, no warmup. Loss spikes to NaN within 200 steps. List three independent fixes, ranked by how much they treat the *cause*. **Answer:** (1) Switch to pre-LN — addresses the underlying gradient instability that post-LN exhibits at depth. (2) Add LR warmup (e.g., 4000 steps) — gives Adam's $\hat v_t$ time to stabilize. (3) Lower peak LR — reduces step magnitude but doesn't address why early steps are unstable. Most causal: switch to pre-LN.

15. **(advanced / construction / `residual-init-scaling`)** A 24-block transformer uses two residual additions per block (attention + MLP). Following GPT-2's prescription, what std should the *output projections* of each sub-block be initialized with, given a base std of 0.02? Sh arithmetic. **Answer:** $N = 24 \times 2 = 48$ residual layers. Std = $0.02 / \sqrt{N} = 0.02/\sqrt{48} \approx 0.00289$. (Or, using nanoGPT's convention, $0.02/\sqrt{2 \cdot 24} = 0.02/\sqrt{48}$, identical.)

16. **(advanced / interpretation / `bn-as-landscape-smoother`)** Santurkar et al. (2018) showed BN's main benefit is *not* reducing internal covariate shift. What did they find it actually does, and what is one experimental finding that supports this? **Answer:** BN makes the loss landscape smoother — it tightens Lipschitz bounds on the loss and on the gradient (i.e., gradients become more predictive). Supporting finding: artificially injecting non-stationary noise *after* BN (so distributions still shift) does not hurt training, contradicting the covariate-shift story.

17. **(advanced / debugging / `loss-curve-pathologies`, `train-val-test-split`)** A team reports test accuracy 92% on a benchmark. After publication, they realize they used the test set to tune the LR schedule across 50 runs. Quantatively, why is their reported number unreliable? **Answer:** The test set was implicitly used as a validation set 50 times; with that many comparisons, the maximum is biased upward by an amount comparable to the standard error of the noise across runs. The 92% number is no longer an unbiased estimate of generalization — it's the maximum over 50 noisy estimates, statistically equivalent to picking the lucky run. The remedy: lock the schedule on a held-out validation set before ever touching the test set.

18. **(advanced / proof-scaffold / `gaussian-prior-mle`, `l2-weight-decay`)** Starting from $w_{\text{MAP}} = \arg\max_w p(w \mid \mathcal{D})$ and the prior $w \sim \mathcal{N}(0, \sigma^2 I)$, derive the equivalent regularized loss and identify $\lambda$ in terms of $\sigma^2$. **Answer:** $p(w|\mathcal{D}) \propto p(\mathcal{D}|w)p(w)$. Take $-\log$: $-\log p(w|\mathcal{D}) = -\log p(\mathcal{D}|w) - \log p(w) + C$. With $-\log p(w) = \frac{1}{2\sigma^2}\|w\|_2^2 + C'$, the objective becomes $-\log pmathcal{D}|w) + \frac{1}{2\sigma^2}\|w\|_2^2$. Comparing to $\mathcal{L}_{\text{data}} + \frac{\lambda}{2}\|w\|_2^2$ gives $\lambda = 1/\sigma^2$.

19. **(advanced / interpretation / `dropout`, `l2-weight-decay`)** Dropout and L2 weight decay both regularize. Give one concrete behavior where they differ. **Answer:** L2 shrinks all weights uniformly; dropout's effective regularization is *adaptive* — it penalizes co-adapted features (where a unit relies on another to compensate) more than independently useful ones. Equivalently, dropout in linear models corresponds to a *data-dependent* L2 penalty proportional to the variance of the inputs, while plain L2 is data-independent.

20. **(advanced / debugging / `gradient-clipping`, `weight-init-symmetry`, `residual-connection`)** A 30-layer plain (no-residual) RNN is trained without gradient clipping; loss diverges within 100 steps. The team adds gradient clipping at norm 1.0 and now the loss is finite but stuck on a plateau forever. Diagnose what is happening  three levels — symptom, immediate cause, and root cause — and propose a fix at each level. **Answer:** *Symptom:* training stalls. *Immediate cause:* clipping is masking exploding gradients but the underlying signal is now squashed below the noise floor; the loss surface still has cliffs. *Root cause:* a 30-layer plain RNN has gradient products with eigenvalues > 1 (or < 1) over many timesteps; vanilla architectures cannot transport gradient through this depth. *Fixes by level:* (symptom) raise clip threshold or LR — minor; (immediate) better init, e.g., orthogonal recurrent matrices — partial; (root) replace with a gated architecture (LSTM/GRU) or add residual/skip connections — addresses the structural problem.

---

## 8. Endgame callback — refined

**Three candidate one-liners**, in increasing specificity:

**A.** *"Open the transformer block you'll train next module: every line is a trick from this module. The `LayerNorm(x)` you're about to write is Lesson 13.4. The `x = x + attn(...)` is  highway from 13.5. The `0.02/√(2·N_layers)` you'll see in the init function is the residual-stream variance argument from 13.3. None of these are optional polish. Without them, deep transformers don't train."*

**B.** *"The transformer block you're about to build is `x = x + F(LN(x))` repeated N times. That `+ x` is the only reason your gradient survives 12 layers; that `LN(x)` is the only reason your activations don't blow up; the `0.02/√(2N)` init on F's output projection is the only reason your residual stream's variance doesn't grow with depth; and the linear warmup over the first 200 steps is the only reason AdamW doesn't take a wild first step into the void. Pre-LN, identity-shortcut, scaled init, warmup — that's a sentence summary of your training setup."* **(Recommended.)**

**C.** *"In Module 14 you'll build sequence models — bigrams, n-grams, RNNs — and watch RNN gradients vanish or explode. The transformer's solution is everything in this module, applied surgically: pre-LN sits insideal stream, residual output projections are initialized at 0.02/√(2·N_layers), and AdamW gets linearly warmed up. We'll write each one of these lines and run it on WebGPU. You'll see the loss curves you learned to read in 13.1, in your browser, in real time."*

---

## 9. Sources (licensing-aware)

1. **Andrej Karpathy, "Building makemore Part 3: Activations & Gradients, BatchNorm"** — YouTube + notebooks at https://github.com/karpathy/nn-zero-to-hero (notebooks MIT-licensed; video itself standard YouTube ToS). Medium: video + Jupyter. **License:** notebooks **MIT [ADAPT]**; video itself **[REFERENCE-ONLY]**. **Use for:** the entire pedagogical scaffolding of Lesson 13.3 (activation/gradient histograms at each depth, the saturated-tanh demo, the 5/3 gain factor). The closest existing match to our pedagogical model.

2. **Karpathy, "A Recipe for Training Neural Networks"** — http://karpathy.github.io/2019/04/25/recipe/. Medium: blog. **License:** unspecified personal blog → **[REFERENCE-ONLY]**. **Usdebugging/diagnosis vocabulary in Lesson 13.1 and the problem bank's debugging-tier problems. Foundational reading for Lesson 13.1's hook.

3. **Distill.pub, Gabriel Goh, "Why Momentum Really Works"** — https://distill.pub/2017/momentum/. Medium: interactive article. **License:** **CC-BY 4.0 [ADAPT]**. **Use for:** visualization-design inspiration for the warmup/lr widgets and any optimizer-callbacks. Reference example for our explorable-explanation tone.

4. **Stanford CS231n notes** — https://cs231n.github.io/neural-networks-2/ (preprocessing, init, regularization, BN). Medium: course notes. **License:** course material, no clear open license (notes attributed but not CC-tagged) → **[REFERENCE-ONLY]**. **Use for:** problem-bank inspiration and the diagnostic-walkthrough vocabulary in Lessons 13.3–13.4. Stanford CS231n slides for spring 2024 are CC-BY 2.0 per the deck cover, but text notes vary; treat as reference-only unless individually verified.

5. **Original primary papers** — Ioffe & SzegedtchNorm, https://arxiv.org/abs/1502.03167); Ba/Kiros/Hinton 2016 (LayerNorm, https://arxiv.org/abs/1607.06450); Srivastava et al. 2014 (Dropout, https://www.jmlr.org/papers/v15/srivastava14a.html); He et al. 2015 (ResNet, https://arxiv.org/abs/1512.03385) and (He init, https://arxiv.org/abs/1502.01852); Glorot & Bengio 2010 (Xavier, http://proceedings.mlr.press/v9/glorot10a.html); Kingma & Ba 2014 (Adam, https://arxiv.org/abs/1412.6980); Loshchilov & Hutter 2017 (AdamW, https://arxiv.org/abs/1711.05101); Pascanu/Mikolov/Bengio 2013 (gradient clipping, https://arxiv.org/abs/1211.5063); Santurkar et al. 2018 (How Does BN Help, https://arxiv.org/abs/1805.11604); Xiong et al. 2020 (pre-LN vs post-LN, https://arxiv.org/abs/2002.04745); Kosson et al. 2024 (warmup analysis, https://arxiv.org/abs/2410.23922). Medium: papers. **License:** standard arXiv preprint terms — citation OK; figure reproduction requires per-paper checking → **[REFERENCE-ONLY]** for figure embedding. **Use for:** authoritative citations, -checking the claims, deriving the formulas in §5.

6. **Goodfellow, Bengio, Courville, *Deep Learning* (MIT Press, 2016)** — https://www.deeplearningbook.org/. Medium: textbook (HTML free; print under MIT Press copyright). **License:** all-rights-reserved textbook → **[REFERENCE-ONLY]**. **Use for:** Chapter 7 (regularization) and Chapter 8 (optimization) for cross-checking definitions and exposition; do not adapt prose or figures.

7. **fast.ai, Practical Deep Learning for Coders** — https://course.fast.ai. Medium: video lectures + notebooks. **License:** library is Apache-2.0 [ADAPT for code]; *course materials* are Apache-licensed in the `course-v3` repo, but lecture videos are personal IP of the authors — **[REFERENCE-ONLY]** for video content. **Use for:** ordering of topics in Lesson 13.1 (their "validation set obsession" pedagogy is the closest model) and the LR-finder framing for Lesson 13.5.

8. **deeplearning.ai, "AI Notes: Initializing neural networks"** — https://www.deeplearning.ai/itialization/. Medium: interactive article. **License:** proprietary deeplearning.ai content → **[REFERENCE-ONLY]**. **Use for:** widget-design inspiration for `varianceFanOut` (their slider-on-init demo is the canonical reference) and for the symmetry-breaking explanation.

9. **TensorFlow Playground** — http://playground.tensorflow.org/, source at https://github.com/tensorflow/playground. Medium: in-browser widget. **License:** **Apache 2.0 [ADAPT]**. **Use for:** code-level reference for our `varianceFanOut` and `lossCurveDoctor` widgets — Apache 2.0 lets us study and reuse with attribution.

10. **MLU-Explain (Amazon), "Bias-Variance Tradeoff," "Train/Test/Val," "Double Descent"** — https://mlu-explain.github.io/. Medium: interactive articles. **License:** text/diagrams **CC-BY-SA-4.0 → [REFERENCE-ONLY]** (share-alike incompatible with commercial product); sample code MIT-0 [ADAPT]. **Use for:** visual-design inspiration for `lossCurveDoctor` and the bias-variance hook in Lesson 13.1; reuse cos but rewrite all prose and visuals.

---

## 10. Pedagogical traps

1. **Trap: presenting BatchNorm before the learner has felt the pain it solves.** The naïve order is: define BN, derive the formula, demo on MNIST. Adult learners then experience BN as bureaucratic notation. *Why it happens:* textbook order treats topics by category, not by motivation. *Mitigation:* in Lesson 13.4, open with the activation-distribution-collapse demo from Lesson 13.3 — BN arrives as the answer to a problem the learner has already watched go wrong on their screen.

2. **Trap: teaching tricks as a list of named techniques rather than as responses to specific failure modes.** The "list of regularizations" lesson — L2, L1, dropout, BN, early stopping, data augmentation, weight decay — is a bibliographic survey, not a learning experience. *Why:* instructors want to be comprehensive and the names are visible. *Mitigation:* every lesson opens with a *failure*, then introduces the trick as the response. Lesson 13.2 opens wititting curves; Lesson 13.3 opens with a dead network; Lesson 13.5 opens with the ResNet degradation problem.

3. **Trap: conflating "regularization" the umbrella term with "L2" the specific technique.** Adult learners with Andrew Ng's ML class background often equate "regularization" with "λ‖w‖²." Then dropout, augmentation, and BN are mysteriously also called regularization. *Why:* one-token meaning collision. *Mitigation:* explicitly define regularization in Lesson 13.2 step 1 as "any modification that biases the search toward simpler hypotheses or constrains the hypothesis space," and tag each subsequent technique with that frame.

4. **Trap: treating warmup as folklore.** "Just use warmup, transformers need it." This is the worst pedagogical move because the learner has no falsification: they can't predict whether warmup is needed for their *next* problem. *Why:* warmup's mechanism is genuinely subtle (high-variance $\hat v$ in Adam, plus correlated early gradients) and the field hasn't fully unif. *Mitigation:* In Lesson 13.5, present warmup *after* showing the post-LN failure live in `warmupStabilizer`. Frame as "Adam's second-moment estimate is unreliable for the first ~200 steps, and pre-LN+warmup are two solutions to the same problem of large early updates." Cite Kosson et al. 2024 explicitly for the SNR argument.

5. **Trap: dumping the 2015 ResNet paper figure without scaffolding why depth was failing in the first place.** The famous "56-layer net is worse than 20-layer net" plot is meaningless without a learner who *expected* depth to help. *Why:* instructors have internalized the surprise; learners haven't. *Mitigation:* In Lesson 13.5 step 1, first force the learner to *predict* what would happen if you stacked more layers, then show the surprise, then introduce residuals. Use `residualGradientFlow` to make the failure mode (vanishing gradient at depth 30, no residual) visible *before* showing the fix.

6. **Trap: showing BN and LN as two algorithms instead of as one operation parameterized by "which axes do I reduce over."** Learners memorize two separate procedures and then can't generalize to GroupNorm, InstanceNorm, RMSNorm. *Why:* the original papers use different notation conventions. *Mitigation:* in Lesson 13.4, after BN and LN are both defined, end with a single tensor of shape (B, C, H, W) and have the learner *predict* which axes BN, LN, GN, IN reduce over. Make the taxonomy explicit. This is a 90-second step that pays back in every future normalization layer they meet.
