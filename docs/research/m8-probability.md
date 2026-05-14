# Module 8 — Probability & Statistics: Pedagogical Research Brief

*Arc 1 (Prerequisite Math) of "Tinker: Machine Learning, Backpropagation, and AI — The Math." Targets adult technical learners (SWE / engineer / scientist). Closest pedagogical model: 3Blue1Brown visual rigor × Karpathy build-from-scratch concreteness. Output of this module is read by Module 9 (Information Theory), which renames negative log-likelihood as cross-entropy.*

---

## 1. Concept Dependency Graph

Topologically sorted (25 concepts). `m2`, `m5`, `m7` refer to earlier course modules.

1. **`sample-space-event`** — The set $\Omega$ of all outcomes of an experiment, together with the notion of an event $A\subseteq\Omega$ as "the thing whose probability we want." *Prereqs:* m2-algebra (sets).
2. **`probability-measure`** — A function $P:2^\Omega\to[0,1]$ satisfying non-negativity, $P(\Omega)=1$, and countable additivity for disjoint events. *Prereqs:* `sample-space-event`.
3. **`random-variable`** — A function $X:\Omega\to\mathbb{R}$ that turns outcomes into numbers we can do math with. *Prereqs:* `probability-measure`, m2-algebra (functions).
4. **`pmf`** — Probability mass function for a discrete $X$: $p_X(x)=P(X=x)$, non-negative, sums to 1. *Prereqs:* `random-variable`.
5. **`pdf`** — Probability density function for a continuous $X$: $f_X$ is a density (not a probability), $\int f_X=1$, and $P(a\le X\le b)=\int_a^b f_X$. *Prereqs:* `pmf`, m5-calculus (integral as area).
6. **`bernoulli`** — $\mathrm{Bern}(p)$: a single biased coin flip, $P(X=1)=p$. *Prereqs:* `pmf`.
7. **`categorical`** — A distribution over $K$ labels with probability vector $(p_1,\dots,p_K)$, $\sum_k p_k=1$. The discrete uniform is the special case $p_k=1/K$. *Prereqs:* `pmf`, m7-linear-algebra (vectors).
8. **`gaussian`** — $\mathcal{N}(\mu,\sigma^2)$: the bell-curve density $\frac{1}{\sigma\sqrt{2\pi}}\exp(-\tfrac{(x-\mu)^2}{2\sigma^2})$. *Prereqs:* `pdf`, m5-calculus (exp).
9. **`joint`** — $p_{X,Y}(x,y)$: probability that $X$ and $Y$ take values $(x,y)$. *Prereqs:* `pmf`, `pdf`.
10. **`marginal`** — Recover one variable's distribution by summing/integrating out the other: $p_X(x)=\sum_y p_{X,Y}(x,y)$. *Prereqs:* `joint`.
11. **`conditional`** — $P(Y\mid X)=P(X,Y)/P(X)$: renormalize the joint to a slice. *Prereqs:* `joint`, `marginal`.
12. **`independence`** — $X\perp Y \iff p_{X,Y}(x,y)=p_X(x)p_Y(y)$. Conditional independence $X\perp Y\mid Z$ is the natural extension. *Prereqs:* `conditional`.
13. **`chain-rule-prob`** — $P(X_1,\dots,X_n)=\prod_i P(X_i\mid X_{<i})$; the factorization that justifies autoregressive language modeling. *Prereqs:* `conditional`.
14. **`bayes-rule`** — $P(\theta\mid\mathcal{D})\propto P(\mathcal{D}\mid\theta)P(\theta)$: posterior $\propto$ likelihood $\times$ prior; framing for generative vs discriminative models. *Prereqs:* `conditional`.
15. **`expectation`** — $\mathbb{E}[X]=\sum_x x\,p(x)$ or $\int x\,f(x)\,dx$; long-run probability-weighted mean. *Prereqs:* `pmf`, `pdf`.
16. **`variance-std`** — $\mathrm{Var}(X)=\mathbb{E}[(X-\mu)^2]$, $\sigma=\sqrt{\mathrm{Var}(X)}$. *Prereqs:* `expectation`.
17. **`covariance`** — $\mathrm{Cov}(X,Y)=\mathbb{E}[(X-\mu_X)(Y-\mu_Y)]$; a dot product of centered random vectors. *Prereqs:* `expectation`, m7-linear-algebra (dot product).
18. **`sample-estimators`** — $\bar x=\tfrac{1}{n}\sum x_i$, $s^2=\tfrac{1}{n-1}\sum(x_i-\bar x)^2$ as estimates of population $\mu,\sigma^2$. *Prereqs:* `expectation`, `variance-std`.
19. **`lln`** — Law of large numbers: $\bar X_n\to\mathbb{E}[X]$ as $n\to\infty$ (one-line intuition only). *Prereqs:* `sample-estimators`.
20. **`clt`** — Central limit theorem: properly scaled sample mean of i.i.d. variables is approximately Gaussian (one-line, calibration only). *Prereqs:* `gaussian`, `lln`.
21. **`mle`** — Maximum likelihood estimation: $\hat\theta=\arg\max_\theta\prod_i P(x_i\mid\theta)=\arg\min_\theta-\sum_i\log P(x_i\mid\theta)$. Foreshadows m9's renaming of NLL as cross-entropy. *Prereqs:* `chain-rule-prob` (i.i.d.), m5-calculus (log/exp).
22. **`sample-categorical`** — Inverse-CDF sampling from a categorical: draw $u\sim\mathrm{Unif}(0,1)$, return the first index where the cumulative sum exceeds $u$. *Prereqs:* `categorical`.
23. **`sample-gaussian`** — Box-Muller intuition: two uniforms $\to$ two independent standard normals (no derivation); affine transform gives any $\mathcal{N}(\mu,\sigma^2)$. *Prereqs:* `gaussian`.
24. **`bias-variance`** — A predictor's expected squared error decomposes into $(\text{bias})^2+\text{variance}+\text{irreducible noise}$; forward pointer to m13. *Prereqs:* `expectation`, `variance-std`.
25. **`autoregressive-lm-bridge`** — A language model is a parametric estimator of $P(x_i\mid x_{<i})$; training $=$ MLE; generation $=$ inverse-CDF sampling from a categorical. The module's payoff concept. *Prereqs:* `chain-rule-prob`, `mle`, `sample-categorical`. → m9-information-theory, m18-capstone.

---

## 2. Canonical Worked Examples

### 2.1 The two-dice joint and marginal
**Problem.** Roll two fair six-sided dice. Let $S=D_1+D_2$, $M=\max(D_1,D_2)$. Compute $P(S=8,M=5)$, $P(M=5)$, and $P(S=8\mid M=5)$.
**Solution.** The 36 equally-likely pairs make up $\Omega$. The event $\{S=8,M=5\}$ is $\{(3,5),(5,3)\}$, so $P=2/36$. The event $\{M=5\}$ is the 25 pairs with both entries $\le 5$ minus the 16 pairs with both $\le 4$: $25-16=9$, so $P(M=5)=9/36$. Then $P(S=8\mid M=5)=(2/36)/(9/36)=2/9$.
**Pedagogical point.** Conditional probability is *renormalization on a slice* — we are not computing in a different universe, we are taking a fixed slice and rescaling it to sum to 1.
**Common mistake.** Trying to compute $P(S=8\mid M=5)$ by counting "ways to get 8 if the max is 5" without first asking what universe they live in. Counter: always write the slice as a literal set first.

### 2.2 The disease-test Bayes problem (3Blue1Brown / Stanford / every prob course)
**Problem.** A disease has prevalence $0.1\%$. A test has $99\%$ sensitivity ($P(+\mid\text{sick})=0.99$) and $99\%$ specificity ($P(-\mid\text{well})=0.99$). You test positive. What is $P(\text{sick}\mid +)$?
**Solution.** By Bayes:
$$P(\text{sick}\mid+)=\frac{0.99\cdot 0.001}{0.99\cdot 0.001+0.01\cdot 0.999}=\frac{0.00099}{0.01098}\approx 0.090.$$
About 9%.
**Pedagogical point.** Posteriors live downstream of priors. Even very accurate tests are dominated by base rates when the base rate is small.
**Common mistake.** Reporting "99%" because the test is "99% accurate." Counter: draw the natural-frequency tree on 100,000 people (3B1B's exact framing).

### 2.3 MLE for a Bernoulli (the bigram prelude)
**Problem.** Observe $n_1$ heads and $n_0$ tails in $n=n_0+n_1$ flips. Find $\hat p$ that maximizes the likelihood under $\mathrm{Bern}(p)$.
**Solution.** Likelihood: $L(p)=p^{n_1}(1-p)^{n_0}$. Log-likelihood: $\ell(p)=n_1\log p+n_0\log(1-p)$. Set $\ell'(p)=0$: $n_1/p-n_0/(1-p)=0\Rightarrow\hat p=n_1/n$.
**Pedagogical point.** The "obvious" estimator (empirical frequency) is the MLE. This is the *exact same move* Karpathy makes when he normalizes the bigram count matrix in makemore.
**Common mistake.** Maximizing $L$ directly and getting lost in products. Counter: always take logs first — products turn into sums (we already taught $\log(ab)=\log a+\log b$ in m5).

### 2.4 Bigram MLE on a tiny corpus (Karpathy-flavored)
**Problem.** Train a bigram model on the single "sentence" `^abab$` (with `^` start and `$` stop tokens). Estimate $P(\text{next}\mid\text{prev}=a)$.
**Solution.** Bigrams: `(^,a), (a,b), (b,a), (a,b), (b,$)`. From `a`, we see `b` twice and nothing else $\Rightarrow\hat P(b\mid a)=1$. From `b`, we see `a` once and `$` once $\Rightarrow\hat P(a\mid b)=\hat P(\$\mid b)=1/2$.
**Pedagogical point.** Count-and-normalize *is* MLE under a categorical (one categorical per context row). This is the bridge from probability to a language model.
**Common mistake.** Forgetting that probabilities exactly 0 produce $\log 0=-\infty$ NLL — motivation for the smoothing/embedding fix in m9 and m10.

### 2.5 Expectation of a categorical / lottery ticket
**Problem.** A ticket costs \$1. With probability $10^{-7}$ you win \$1,000,000; otherwise \$0. What is $\mathbb{E}[\text{net}]$?
**Solution.** $\mathbb{E}[\text{net}]=10^{-7}\cdot 999{,}999-(1-10^{-7})\cdot 1\approx -0.9$.
**Pedagogical point.** Expectation is the probability-weighted sum, *not* the modal outcome. Almost every individual realization is $-\$1$; the average is what scales with $n$.
**Common mistake.** Confusing "what will happen" with "average over many trials." Counter: simulate 10,000 trials in the widget and watch $\bar x$ approach $\mathbb{E}[X]$.

### 2.6 Variance of a sum of independents (the BatchNorm / init prelude)
**Problem.** Let $X_1,\dots,X_n$ be i.i.d. with mean 0 and variance $\sigma^2$. What are $\mathbb{E}[S_n]$ and $\mathrm{Var}(S_n)$ where $S_n=\sum X_i$? What about $\bar X_n=S_n/n$?
**Solution.** $\mathbb{E}[S_n]=0$. $\mathrm{Var}(S_n)=n\sigma^2$ (variances add for independents). $\mathrm{Var}(\bar X_n)=\sigma^2/n$.
**Pedagogical point.** Variances add under independence; standard deviations don't. This is the exact reason He/Xavier initialization scales weights by $1/\sqrt{\text{fan-in}}$ — keep variance constant as signal flows through a layer.
**Common mistake.** Saying $\sigma_{S_n}=n\sigma$ instead of $\sqrt{n}\sigma$. Counter: simulate $n=100$ Bernoulli flips, plot histogram, overlay theoretical $\sigma$.

### 2.7 Covariance as a dot product of centered vectors (the linear-algebra bridge)
**Problem.** Given paired observations $\{(x_i,y_i)\}_{i=1}^n$, show that sample covariance $\hat{\mathrm{Cov}}(X,Y)=\tfrac{1}{n-1}\sum(x_i-\bar x)(y_i-\bar y)$ equals $\tfrac{1}{n-1}\langle\tilde x,\tilde y\rangle$ where $\tilde x_i=x_i-\bar x$.
**Solution.** Direct expansion: the sum-of-products *is* the inner product of the centered vectors. Then Pearson correlation $=\tilde x\cdot\tilde y/(\|\tilde x\|\|\tilde y\|)=\cos\theta$.
**Pedagogical point.** Covariance is "dot product of centered data," correlation is "cosine similarity of centered data." This collapses a huge amount of seemingly separate machinery into one picture and pays back to m7.
**Common mistake.** Treating covariance as an opaque formula. Counter: drag a point in a scatterplot and watch both $\tilde x\cdot\tilde y$ and $\cos\theta$ update.

### 2.8 Sampling from a categorical via cumulative sum (the m18 sampler verbatim)
**Problem.** Given probabilities $\mathbf{p}=(0.1,0.5,0.4)$, write pseudocode that returns index 0, 1, or 2 with the right frequencies using a single $u\sim\mathrm{Unif}(0,1)$.
**Solution.** Compute CDF $c=(0.1,0.6,1.0)$. Draw $u$. Return the smallest $i$ with $c_i\ge u$.
```js
function sampleCategorical(p) {
  const u = Math.random();
  let cum = 0;
  for (let i = 0; i < p.length; i++) {
    cum += p[i];
    if (cum >= u) return i;
  }
  return p.length - 1; // numeric safety
}
```
**Pedagogical point.** This is the *literal* sampler the capstone uses. We do not need a black box; we just invert the CDF.
**Common mistake.** Using `<` vs `<=` inconsistently, or forgetting the numeric-safety fallthrough when probabilities don't sum to exactly 1.

### 2.9 Standardizing a Gaussian (the weight-init derivation)
**Problem.** If $X\sim\mathcal{N}(\mu,\sigma^2)$ and $Z=(X-\mu)/\sigma$, what is the distribution of $Z$? If $X\sim\mathcal{N}(0,1)$ and $Y=\sigma X+\mu$, what is the distribution of $Y$?
**Solution.** $Z\sim\mathcal{N}(0,1)$. $Y\sim\mathcal{N}(\mu,\sigma^2)$. Affine transforms keep Gaussians Gaussian; the parameters just shift.
**Pedagogical point.** "Multiply by $\sigma$, add $\mu$" is *how* `torch.randn` becomes a weight tensor with the variance you want. It is also exactly what BatchNorm does at inference time, with learned $\mu,\sigma$.
**Common mistake.** "Converting" by adding $\sigma^2$ instead of multiplying by $\sigma$. Counter: a draggable handle that scales then shifts a sampled cloud.

### 2.10 The mode is not the mean — categorical sampling vs argmax
**Problem.** A trained LM outputs logits $(2.0,1.5,1.3,0.1,0.1)$ for the next token. Compute the softmax, then describe how "greedy decoding" differs from "sampling."
**Solution.** Softmax: subtract max for stability $\to(0,-0.5,-0.7,-1.9,-1.9)$, exp $\to(1,0.607,0.497,0.150,0.150)$, normalize $\to\approx(0.418,0.254,0.207,0.062,0.062)$. Argmax picks token 0 every time. Sampling returns token 0 about 42% of the time, token 1 about 25%, etc.
**Pedagogical point.** "Greedy" is $\arg\max p$; "sampling" is drawing from the categorical. Both come from the same probability vector — they are different *post-processing* steps. Temperature (m9) shapes the vector before either step.
**Common mistake.** Believing the model "knows" the answer and sampling is somehow noisier. Counter: a widget exposing the same probability bar chart for both modes.

---

## 3. Common Misconceptions

1. **"Probability density is probability."**
   *Why natural:* For pmf, $p(x)$ *is* the probability of $X=x$. The names look identical.
   *Kill:* Show a uniform $\mathrm{Unif}(0,0.5)$ — its density is 2, which would be "impossible" if density were probability. Density is "probability per unit $x$"; only the area is probability.

2. **"$P(A\mid B)=P(B\mid A)$."**
   *Why natural:* English makes "given" feel symmetric. The prosecutor's fallacy lives here.
   *Kill:* "$P(\text{positive}\mid\text{sick})=0.99$ does not mean $P(\text{sick}\mid\text{positive})=0.99$." Drive home with the disease-test example.

3. **"Independent and mutually exclusive are the same thing."**
   *Why natural:* Both feel like "not connected."
   *Kill:* If $A$ and $B$ are mutually exclusive with $P(A),P(B)>0$, then $P(A\cap B)=0\ne P(A)P(B)$, so they are emphatically *dependent* — knowing $A$ happened tells you $B$ did not.

4. **"The Gaussian is special because the CLT says everything is Gaussian."**
   *Why natural:* Pop-science telephone.
   *Kill:* The CLT is about *sums/means* of many i.i.d. terms with finite variance. Individual data — incomes, token frequencies, file sizes — are usually wildly non-Gaussian (power-law). Show a histogram of token frequencies in a real corpus.

5. **"Maximizing likelihood and minimizing error are unrelated."**
   *Why natural:* Different vocabulary, different chapters in different textbooks.
   *Kill:* For Gaussian noise, MLE *is* least squares. For Bernoulli/Categorical labels, MLE *is* cross-entropy. Show the derivation once, then label every future loss with both names side-by-side.

6. **"A 95% confidence interval has 95% probability of containing the true parameter."**
   *Why natural:* It's how every newspaper paraphrases it.
   *Kill:* Under frequentist semantics, $\theta$ is fixed; the *interval* is random. The 95% refers to the procedure's long-run coverage. State it correctly once and move on — full inference is out of scope.

7. **"$\mathbb{E}[XY]=\mathbb{E}[X]\mathbb{E}[Y]$ always."**
   *Why natural:* Linearity of expectation makes everything look multiplicative.
   *Kill:* This only holds under independence. Counterexample: $Y=X$ with $X\in\{-1,+1\}$ uniform — $\mathbb{E}[X]=0$ so product of means is 0, but $\mathbb{E}[X^2]=1$.

8. **"Bayes' theorem requires Bayesian philosophy."**
   *Why natural:* The word "Bayesian" is loaded with culture-war baggage.
   *Kill:* Bayes' theorem is *an algebraic identity* derivable in one line from the definition of conditional probability. Frequentists use it daily for the disease-test calculation. The philosophy is about whether $\theta$ has a prior, not whether the theorem is true.

9. **"$\log P$ is negative, which is bad."**
   *Why natural:* Engineering background says positive numbers are good.
   *Kill:* For $P\in(0,1)$, $\log P<0$, so $-\log P>0$ is a positive "surprise" or "cost." NLL is the sum of these surprises. This sentence is the bridge to m9-entropy.

10. **"The sample mean equals the population mean."**
    *Why natural:* In the limit it does (LLN).
    *Kill:* Roll 10 dice and compute $\bar x$. Show how it bounces around 3.5. Tighten as $n$ grows. This is also the LLN demo.

11. **"Sampling from a model with low temperature gives 'better' samples."**
    *Why natural:* The slider is often labeled "0 = best."
    *Kill:* Temperature $\to 0$ collapses to argmax. Argmax is *one mode* — it ignores the rest of the distribution and gives repetitive, low-entropy output. Show a side-by-side widget; the right temperature is task-dependent.

12. **"i.i.d. is a technicality."**
    *Why natural:* It is buried in fine print of every theorem.
    *Kill:* Show what happens to MLE if you give it 10 copies of the same data point: $\hat p$ doesn't change, but your *confidence* about it pretends to. The chain-rule factorization $\prod P(x_i\mid\theta)$ assumes i.i.d.; language models break this on purpose (the chain rule of probability still holds, but tokens are not independent — that's what attention is for).

---

## 4. Interactive Widget Suggestions

All widgets target Svelte 5 + Mafs + KaTeX. Each lets the learner manipulate a named mathematical object directly.

### 4.1 `bayesGrid`
- **Direct manipulation:** Drag the four corners of a 100×100 outcome grid: prevalence, sensitivity, specificity (and total population). Click cells to toggle "what if this person is sick / tests positive."
- **Live updates:** The four-quadrant population partition (TP/FP/FN/TN) animates; $P(\text{sick}\mid +)$ updates in big type; the Bayes' theorem equation has each numeric term highlighted as you drag.
- **Pedagogical moment:** Posterior $\ne$ test accuracy. The viewer *sees* false positives swallow true positives when prevalence is tiny.
- **Why it beats a slider:** Sliders force one-at-a-time mental updates. The grid is the natural-frequency representation that 3B1B and Gigerenzer both argue is how humans actually understand Bayes. Dragging a corner reorganizes the population, not just a number.
- **Prior art:** 3Blue1Brown's "Bayes theorem" video (https://www.youtube.com/watch?v=HZGCoVF3YvM) [REFERENCE-ONLY]; Setosa conditional probability (https://setosa.io/conditional/) [REFERENCE-ONLY]; Seeing Theory's "Compound Probability" page (https://seeing-theory.brown.edu/compound-probability/index.html) [REFERENCE-ONLY].

### 4.2 `cdfSampler`
- **Direct manipulation:** Drag the top edges of bars in a categorical bar chart over a vocabulary `[a,b,c,d,e]` to set $\mathbf{p}$ (auto-renormalized). Click a "drop a ball" button to inject a uniform $u$ that visibly falls onto a stacked CDF bar and labels which token wins.
- **Live updates:** Bars on the left; the cumulative-sum staircase $c_0,c_1,\dots,c_K$ on the right; the ball's $u$-value as a horizontal line; the chosen token labeled below; a running tally of the last 200 samples as a histogram converging to $\mathbf{p}$.
- **Pedagogical moment:** "Sampling from a categorical" is not magic — it is *exactly* picking which interval on $[0,1]$ a uniform random number lands in. This is the literal m18 capstone sampler.
- **Why it beats a slider:** The learner sees the inverse-CDF picture they are about to *code*. They are dragging probabilities, dropping uniforms, watching frequencies converge — three named objects, all touchable.
- **Prior art:** Karpathy's makemore Lecture 1 (https://www.youtube.com/watch?v=PaCmpygFfXo) [REFERENCE-ONLY]; Seeing Theory "Probability Distributions" [REFERENCE-ONLY].

### 4.3 `jointMarginalScrub`
- **Direct manipulation:** Paint heat onto a 5×5 grid representing the joint pmf $p_{X,Y}$ (auto-normalized to sum to 1). Click a row or column to "condition on" it — that row is extracted, renormalized, and shown as a separate bar chart.
- **Live updates:** Top margin bar chart $=p_X$ (sum across columns), right margin bar chart $=p_Y$, central heatmap $=$ joint, plus a popped-out conditional $p(Y\mid X=x)$ when a row is clicked. KaTeX equations beside show what summation is happening on each side.
- **Pedagogical moment:** Marginalization is summing along an axis; conditioning is taking a slice and renormalizing. Both are visible.
- **Why it beats a slider:** A slider can't show that conditional $=$ "slice + renormalize." A 2D painting + slice tool can.
- **Prior art:** Seeing Theory "Probability Distributions" [REFERENCE-ONLY]; Mathematics for Machine Learning §6.2 figures (https://mml-book.github.io/) [REFERENCE-ONLY].

### 4.4 `gaussianForge`
- **Direct manipulation:** Drag the peak of a Gaussian curve to set $\mu$, drag the inflection point to set $\sigma$. Below, click "sample 1," "sample 100," or "sample 10k" — points rain down onto a histogram beneath the curve.
- **Live updates:** $\mu,\sigma$ readouts; the empirical histogram converging to the density (LLN, live); $\bar x$ and $s$ shown alongside true $\mu,\sigma$ to highlight estimator behavior. Toggle: "show standardized $Z=(X-\mu)/\sigma$" snaps the curve to $\mathcal{N}(0,1)$.
- **Pedagogical moment:** Affine transforms of Gaussians are Gaussians. Standardization is just "drag the parameters back to (0,1)." This is *literally* what BatchNorm does.
- **Why it beats a slider:** Dragging the inflection point makes $\sigma$ a geometric object instead of an opaque parameter. The standardization toggle makes the affine map a clickable physical reset.
- **Prior art:** Seeing Theory "Continuous Random Variables" [REFERENCE-ONLY]; Distill "A Visual Exploration of Gaussian Processes" (https://distill.pub/2019/visual-exploration-gaussian-processes/) [ADAPT — CC BY 4.0].

### 4.5 `mleClimber`
- **Direct manipulation:** A coin-flip dataset is shown as a row of H/T glyphs (user can click to flip any of them). A handle on the curve $L(p)=p^{n_1}(1-p)^{n_0}$ moves a vertical line representing the candidate $\hat p$.
- **Live updates:** As the user moves $\hat p$, both $L(p)$ and $\ell(p)=\log L(p)$ update. The product $\prod p(x_i\mid\hat p)$ is rendered factor-by-factor (all terms visible) so the learner sees the product collapse multiplicatively. A "snap to argmax" button jumps to the analytic maximum $n_1/n$.
- **Pedagogical moment:** MLE is *climbing the likelihood curve* and the empirical frequency is the top. Also: products under-/overflow fast; logs save us.
- **Why it beats a slider:** The slider exists, but its job is to make the *product expansion* and the *log curve* both visible simultaneously. The learner sees why log-likelihood is preferred without being lectured.
- **Prior art:** Karpathy makemore part 1 (https://www.youtube.com/watch?v=PaCmpygFfXo) [REFERENCE-ONLY]; StatLect MLE chapter (https://www.statlect.com/fundamentals-of-statistics/maximum-likelihood) [REFERENCE-ONLY].

### 4.6 `covarianceScatter`
- **Direct manipulation:** Drag individual points around a 2D scatterplot.
- **Live updates:** Live readout of $\bar x,\bar y$ (the centroid, drawn as a crosshair); each point shows its centered vector $(x_i-\bar x,y_i-\bar y)$ as a faint arrow from the centroid; the running sum-of-products and the Pearson $r$ update in real time. Overlay: two centered vectors $\tilde x,\tilde y\in\mathbb{R}^n$ in a side panel, with $\cos\theta$ shown geometrically.
- **Pedagogical moment:** "Covariance is a dot product of centered data" is a *picture*, not a definition. Correlation is cosine similarity. This is the bridge to m7.
- **Why it beats a slider:** No slider would let the learner *cause* a high covariance by dragging points into a diagonal line.
- **Prior art:** Seeing Theory "Basic Probability" [REFERENCE-ONLY]; OpenIntro Statistics §7.1 correlation figures [REFERENCE-ONLY — CC BY-SA].

### 4.7 `bigramSampler`
- **Direct manipulation:** A 27×27 heatmap of bigram counts (rows $=$ previous char, cols $=$ next char) from a small fixed corpus of names. Click a row to "condition on" that previous character; the row renders as a categorical bar chart on the right. Click "generate a name" to roll the autoregressive sampler step by step, with each draw using the `cdfSampler` mechanism visually animating.
- **Live updates:** Heatmap, conditional bar chart for selected row, current name being generated character by character, running NLL of the generated string, and a textbox showing the chain-rule decomposition $P(\text{name})=\prod P(c_i\mid c_{i-1})$.
- **Pedagogical moment:** This is the m9/m18 endpoint, *in this module*. Probability isn't an abstraction — it is the exact mechanism that generates text. Karpathy's makemore in 200 lines of Svelte.
- **Why it beats a slider:** The learner is literally clicking the conditional and watching the autoregressive product unfold. They see chain rule of probability, categorical sampling, and MLE-from-counts simultaneously.
- **Prior art:** Karpathy makemore part 1 (the original) [REFERENCE-ONLY for the video; the count-and-normalize *code pattern* in nanoGPT/makemore is MIT-licensed → ADAPT].

---

## 5. Key Formulas (LaTeX-Ready)

### Axioms & basic objects
```
P(A) \ge 0
P(\Omega) = 1
P\!\left(\bigsqcup_i A_i\right) = \sum_i P(A_i)
X : \Omega \to \mathbb{R}
```

### Pmf / pdf
```
\sum_{x} p_X(x) = 1
\int_{-\infty}^{\infty} f_X(x)\, dx = 1
P(a \le X \le b) = \int_a^b f_X(x)\, dx
```

### Canonical distributions
```
\mathrm{Bern}(p):\quad P(X=1)=p,\ P(X=0)=1-p
\mathrm{Cat}(\mathbf{p}):\quad P(X=k)=p_k,\ \sum_k p_k = 1
\mathcal{U}\{1,\dots,K\}:\quad P(X=k)=\tfrac{1}{K}
\mathcal{N}(\mu,\sigma^2):\quad f(x)=\frac{1}{\sigma\sqrt{2\pi}}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
```

### Joint, marginal, conditional, independence
```
p_{X,Y}(x,y) = P(X=x,\ Y=y)
p_X(x) = \sum_{y} p_{X,Y}(x,y) \qquad f_X(x) = \int f_{X,Y}(x,y)\,dy
P(Y\mid X) = \frac{P(X,Y)}{P(X)}
X \perp Y \iff p_{X,Y}(x,y) = p_X(x)\, p_Y(y)
X \perp Y \mid Z \iff p_{X,Y\mid Z}(x,y\mid z) = p_{X\mid Z}(x\mid z)\, p_{Y\mid Z}(y\mid z)
```

### Chain rule and Bayes
```
P(X_1,\dots,X_n) = \prod_{i=1}^{n} P(X_i \mid X_1, \dots, X_{i-1})
P(\theta \mid \mathcal{D}) = \frac{P(\mathcal{D}\mid\theta)\, P(\theta)}{P(\mathcal{D})}
P(\theta \mid \mathcal{D}) \propto P(\mathcal{D}\mid\theta)\, P(\theta)
```

### Expectation, variance, covariance
```
\mathbb{E}[X] = \sum_x x\, p(x) \quad\text{or}\quad \int x\, f(x)\, dx
\mathbb{E}[aX + bY] = a\,\mathbb{E}[X] + b\,\mathbb{E}[Y]
\mathrm{Var}(X) = \mathbb{E}[(X-\mu)^2] = \mathbb{E}[X^2] - \mathbb{E}[X]^2
\sigma_X = \sqrt{\mathrm{Var}(X)}
\mathrm{Cov}(X,Y) = \mathbb{E}[(X-\mu_X)(Y-\mu_Y)]
\mathrm{Var}(aX + bY) = a^2\mathrm{Var}(X) + b^2\mathrm{Var}(Y) + 2ab\,\mathrm{Cov}(X,Y)
\mathrm{Corr}(X,Y) = \frac{\mathrm{Cov}(X,Y)}{\sigma_X \sigma_Y}
```

### Estimators
```
\bar X_n = \frac{1}{n}\sum_{i=1}^n X_i
s^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar X_n)^2
```

### LLN / CLT (one line each)
```
\bar X_n \xrightarrow{n\to\infty} \mathbb{E}[X]
\frac{\bar X_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} \mathcal{N}(0,1)
```

### Maximum likelihood
```
L(\theta) = \prod_{i=1}^n P(x_i \mid \theta)
\ell(\theta) = \sum_{i=1}^n \log P(x_i \mid \theta)
\hat\theta_{\mathrm{MLE}} = \arg\max_{\theta} \ell(\theta) = \arg\min_{\theta}\ \bigl(-\sum_{i=1}^n \log P(x_i\mid\theta)\bigr)
```

### Sampling
```
\text{Inverse-CDF (categorical): } u\sim\mathcal{U}(0,1),\ \ k^* = \min\{k:\ \sum_{j\le k} p_j \ge u\}
\text{Box-Muller: } U_1,U_2\sim\mathcal{U}(0,1) \ \Rightarrow\ Z_1 = \sqrt{-2\ln U_1}\cos(2\pi U_2)\sim\mathcal{N}(0,1)
\text{Affine: } Z\sim\mathcal{N}(0,1) \Rightarrow \sigma Z + \mu \sim \mathcal{N}(\mu,\sigma^2)
```

### Bias–variance (one-page intuition; full in m13)
```
\mathbb{E}\!\left[(y - \hat f(x))^2\right] = \bigl(\mathbb{E}[\hat f(x)] - f(x)\bigr)^2 + \mathrm{Var}(\hat f(x)) + \sigma_{\varepsilon}^2
```

---

## 6. Lesson Decomposition

**Opinionated recommendation:** five lessons, not four and not six. Four crams chain rule and MLE together and loses the punchline. Six adds a "philosophy of probability" lesson that fascinates instructors and confuses students. Cut it.

### Lesson 8.1 — "What does probability *measure*?"
**Summary:** Probability is just a rule that assigns numbers in $[0,1]$ to sets of outcomes, and it has to play nicely with unions and complements. Random variables are how we drag those numbers onto the real line so we can do math.
**Where this shows up in the transformer:** The transformer's final output is a probability vector over the vocabulary; everything in this module is the formal grammar for what that vector even means.
**Steps:**
1. *Hook:* Roll two dice, sum is 7 — how many ways? (Count outcomes in $\Omega$.)
2. *Sample space and event* — the set $\Omega$ and a subset $A\subseteq\Omega$.
3. *The three axioms* — non-negativity, normalization, additivity for disjoint events.
4. **StepCheck:** Given $P(A)=0.6,P(B)=0.5,P(A\cap B)=0.3$, find $P(A\cup B)$. *Expected:* $0.8$.
5. *Random variable as a function* — show $X=D_1+D_2$ literally as a mapping.
6. *Pmf vs pdf* — bars sum to 1; densities integrate to 1; density is not probability.
7. **StepCheck:** A density $f(x)=cx$ on $[0,2]$, zero elsewhere. Find $c$. *Expected:* $1/2$.
8. *Three discrete distributions we will see forever* — Bernoulli, Categorical, Uniform.
9. *The Gaussian* — visual: bell, $\mu,\sigma$ as drag handles.
10. *Forward pointer:* "Every layer of the transformer maps inputs to a categorical at the end. This module tells us what categorical means."
**Widgets:** `cdfSampler` (for showing what a categorical is), `gaussianForge`.
**Estimated minutes:** 35.

### Lesson 8.2 — "Joint, marginal, conditional"
**Summary:** When two random variables share a world, the joint pmf is the whole picture, marginals are projections, and conditioning is taking a slice and renormalizing it.
**Where this shows up in the transformer:** Attention computes conditional distributions $P(\text{next token}\mid\text{context})$; chain rule of probability is what turns these into a sentence.
**Steps:**
1. *Hook:* Two coin flips' joint table (4 cells).
2. *Joint pmf* on a grid.
3. *Marginalize* — sum out a variable.
4. **StepCheck:** Given joint pmf on $\{0,1\}^2$ with values $(0.1,0.4,0.2,0.3)$ (row-major), find $P(X=1)$. *Expected:* $0.5$.
5. *Conditional* as renormalized slice.
6. **StepCheck:** Same table — find $P(Y=1\mid X=1)$. *Expected:* $0.6$.
7. *Independence* — joint factors as product of marginals.
8. *Conditional independence* — $X\perp Y\mid Z$; the assumption naive Bayes uses and attention deliberately *breaks*.
9. *Chain rule of probability* — $P(x_1,\dots,x_n)=\prod_i P(x_i\mid x_{<i})$.
10. *Why chain rule justifies autoregressive LMs* — every LM is a parametric estimator of conditional next-token distributions.
11. **StepCheck:** Using a 4-token sequence with given conditionals, compute the joint. *Expected:* $0.024$.
12. *Bayes' theorem* — flip the conditioning.
13. *Disease-test worked example* with prior $0.001$, sens/spec $0.99$. **StepCheck:** posterior. *Expected:* $\approx 0.090$.
14. *Generative vs discriminative framing* — generative models learn $P(x,y)$, discriminative learn $P(y\mid x)$; both legal, both used.
**Widgets:** `bayesGrid`, `jointMarginalScrub`.
**Estimated minutes:** 45.

### Lesson 8.3 — "What you can say about a random variable in one number"
**Summary:** Mean, variance, covariance — three summary statistics with which we lose almost nothing important about Gaussian models.
**Where this shows up in the transformer:** Weight initialization sets variance scaling; BatchNorm and LayerNorm explicitly compute mean and variance; loss values you see during training are sample means.
**Steps:**
1. *Hook:* Expected value of a lottery ticket.
2. *Expectation* — probability-weighted sum.
3. *Linearity of expectation* — always holds, even for dependent variables.
4. **StepCheck:** $\mathbb{E}[X]$ for the 3-token categorical $(0.1,0.5,0.4)$ over $\{0,1,2\}$. *Expected:* $1.3$.
5. *Variance and standard deviation* — spread.
6. *Variances add under independence* — show $\mathrm{Var}(\bar X_n)=\sigma^2/n$, the formula behind weight init.
7. **StepCheck:** $X\sim\mathrm{Bern}(0.3)$, find $\mathrm{Var}(X)$. *Expected:* $0.21$.
8. *Covariance* as $\mathbb{E}[(X-\mu_X)(Y-\mu_Y)]$.
9. *Covariance as dot product of centered data* — bridge back to m7.
10. *Correlation as cosine similarity* of centered data.
11. *Sample mean and sample variance as estimators* — what changes from $\mathbb{E}$ to $\bar x$.
12. *LLN in one line* — sample mean converges to expectation.
13. *CLT in one line* — sample means are asymptotically Gaussian; this is why error bars are Gaussian even when data aren't.
**Widgets:** `covarianceScatter`, `gaussianForge` (for LLN/CLT live demo).
**Estimated minutes:** 50.

### Lesson 8.4 — "Maximum likelihood: 'fit the data' has a formula"
**Summary:** MLE is the formal version of "pick the parameters that make the data least surprising." For Bernoulli/Categorical, it's count-and-normalize. For Gaussian noise, it's least squares.
**Where this shows up in the transformer:** Every loss function in this course will be either negative log-likelihood (cross-entropy) or a direct alias of it. Training *is* MLE.
**Steps:**
1. *Hook:* You have one biased coin and 10 flips. What's $\hat p$?
2. *Likelihood function* — $L(\theta)=\prod_i P(x_i\mid\theta)$, viewed as a function of $\theta$ with data fixed.
3. *Why take logs* — products underflow; sums don't; argmax is preserved.
4. *MLE for Bernoulli* — full derivation; $\hat p=n_1/n$.
5. **StepCheck:** 7 heads in 10 flips, $\hat p$. *Expected:* $0.7$.
6. *MLE for a Categorical* — generalize: $\hat p_k=n_k/n$.
7. *MLE for a Gaussian* — quick: $\hat\mu=\bar x$, $\hat\sigma^2=\tfrac{1}{n}\sum(x_i-\bar x)^2$ (with $n$, not $n-1$; biased but MLE).
8. *MLE $\to$ least squares* — under Gaussian noise, max-likelihood is least squares. One line, no chapter.
9. *Bigram MLE on a corpus* — count, normalize, get the bigram model. **StepCheck:** for the corpus `^ab$ ^ac$`, compute $\hat P(b\mid a)$. *Expected:* $0.5$.
10. *NLL $=-\sum_i\log P(x_i\mid\theta)$* — the form you will minimize forever.
11. *Forward pointer (the m9 handoff line):* "Next module: this exact quantity is also called cross-entropy. The name 'cross-entropy loss' on a PyTorch training loop refers to this line."
**Widgets:** `mleClimber`, `bigramSampler`.
**Estimated minutes:** 45.

### Lesson 8.5 — "Drawing samples (and what could possibly go wrong)"
**Summary:** Given a trained model, *generating* output means drawing from a probability distribution. We will write the sampler ourselves; it's six lines.
**Where this shows up in the transformer:** This lesson is exactly the m18 capstone's `sample()` function, in advance. Temperature, top-k, top-p (m9, m16) all sit on top of what we build here.
**Steps:**
1. *Hook:* "torch.multinomial seems magic; let's de-magic it."
2. *Sampling a uniform* — every PRNG returns $u\sim\mathcal{U}(0,1)$.
3. *Inverse-CDF sampling from a categorical* — the cumulative-sum trick.
4. *Walking through the algorithm by hand* — $\mathbf{p}=(0.1,0.5,0.4)$ with $u=0.65\to$ return index 1.
5. **StepCheck:** With $\mathbf{p}=(0.2,0.3,0.5)$ and $u=0.45$, which index? *Expected:* $1$.
6. *Implementing it in JavaScript* — the exact 8-line function we'll re-use.
7. *Sampling from a Gaussian: Box-Muller intuition* — two uniforms in, two standard normals out, no derivation.
8. *Affine transform to any $\mathcal{N}(\mu,\sigma^2)$* — multiply by $\sigma$, add $\mu$ (the weight-init move).
9. *Sampling vs argmax* (greedy decoding vs sampling) — same probability vector, different post-processing.
10. *Bias-variance tradeoff, one page* — what changes between sample mean of 10 draws and sample mean of $10^6$ draws? Forward pointer to m13.
11. *Endgame callback* (see Section 8 below).
**Widgets:** `cdfSampler`, `gaussianForge`, `bigramSampler`.
**Estimated minutes:** 40.

**Total module:** ~215 min of guided interactive content (≈ 3.5 hours of focused work).

---

## 7. Problem Bank (20)

Tags refer to concept IDs in §1.

1. **(novice, computation, `probability-measure`)** A fair coin is flipped twice. What is $P(\text{at least one head})$? *Expected:* $3/4$.
2. **(novice, computation, `pmf`)** A categorical RV over $\{a,b,c,d\}$ has $p_a=0.2, p_b=0.3, p_d=0.1$. What is $p_c$? *Expected:* $0.4$.
3. **(novice, computation, `pdf`)** $f(x)=c$ on $[3,7]$, zero elsewhere. Find $c$. *Expected:* $1/4$.
4. **(novice, computation, `expectation`)** $X\in\{-1,0,2\}$ with probabilities $(0.5,0.3,0.2)$. Find $\mathbb{E}[X]$. *Expected:* $-0.1$.
5. **(novice, computation, `variance-std`)** Same $X$ as #4. Find $\mathrm{Var}(X)$. *Expected:* $1.69$.
6. **(novice, interpretation, `bernoulli`)** Write a single sentence describing a real situation well-modeled by $\mathrm{Bern}(0.05)$. *Expected:* any "rare binary event" — e.g., "a particular click on a webpage."
7. **(intermediate, computation, `conditional`)** A bag has 3 red, 7 blue marbles. Draw two without replacement. What is $P(\text{second blue}\mid\text{first red})$? *Expected:* $7/9$.
8. **(intermediate, computation, `bayes-rule`)** Prior $P(\theta=A)=0.2, P(\theta=B)=0.8$. Likelihoods $P(x\mid A)=0.9, P(x\mid B)=0.1$. Find $P(\theta=A\mid x)$. *Expected:* $0.18/0.26\approx 0.692$.
9. **(intermediate, computation, `chain-rule-prob`)** A bigram model has $P(a\mid\hat{})=0.3, P(b\mid a)=0.4, P(\$\mid b)=0.2$. What is $P(\hat{}ab\$)$? *Expected:* $0.024$.
10. **(intermediate, computation, `marginal`)** Joint pmf on $\{0,1\}\times\{0,1\}$: $p(0,0)=0.1, p(0,1)=0.2, p(1,0)=0.3, p(1,1)=0.4$. Compute $p_X(1)$ and $p_Y(0)$. *Expected:* $0.7, 0.4$.
11. **(intermediate, construction, `independence`)** Construct a joint distribution on $\{0,1\}^2$ where $X,Y$ are *not* independent, and another where they *are*, both with $p_X(1)=0.5, p_Y(1)=0.5$. *Expected:* For independence, all four cells $=0.25$; for dependence, any $(p_{00},p_{01},p_{10},p_{11})$ summing to 1 with marginals $0.5,0.5$ but $p_{11}\ne 0.25$, e.g. $(0.4,0.1,0.1,0.4)$.
12. **(intermediate, computation, `gaussian`)** $X\sim\mathcal{N}(3,4)$. What are $\mu,\sigma$? Where are the inflection points of the density? *Expected:* $\mu=3, \sigma=2$; inflections at $x=1$ and $x=5$.
13. **(intermediate, computation, `expectation`)** Two i.i.d. $X_1,X_2\sim\mathrm{Bern}(p)$. Find $\mathbb{E}[X_1+X_2]$ and $\mathrm{Var}(X_1+X_2)$. *Expected:* $2p, 2p(1-p)$.
14. **(intermediate, computation, `covariance`)** Data: $(1,2),(2,4),(3,6)$. Compute the sample covariance (with $1/(n-1)$). *Expected:* $2$.
15. **(intermediate, computation, `mle`)** A six-sided die rolled 60 times produces face-counts $(8,10,12,9,11,10)$. State the MLE for each $p_k$. *Expected:* $(8/60,10/60,12/60,9/60,11/60,10/60)$.
16. **(intermediate, debugging, `pmf`)** A student writes $P(X=k)=k/10$ for $k\in\{1,2,3,4\}$. Is this a valid pmf? If not, fix it so the support is $\{1,2,3,4,5\}$. *Expected:* For $\{1,2,3,4\}$ the sum is $1+2+3+4=10$ over $10$ so it *is* valid; for the extended support, normalize by $15$, giving $P(k)=k/15$.
17. **(advanced, proof-scaffold, `mle`)** Derive the MLE $\hat\mu,\hat\sigma^2$ for $\{x_1,\dots,x_n\}\sim\mathcal{N}(\mu,\sigma^2)$. Show $\hat\mu=\bar x$. *Expected:* $\hat\mu=\bar x$, $\hat\sigma^2=\frac{1}{n}\sum(x_i-\bar x)^2$ (note: biased estimator; biased but MLE — flag explicitly).
18. **(advanced, construction, `sample-categorical`)** Write 8 lines of JavaScript that takes an array `p` of nonnegative numbers summing to 1 and returns a random index sampled according to `p`. *Expected:* implementation matching §2.8.
19. **(advanced, proof-scaffold, `chain-rule-prob`)** Show that the chain rule of probability follows from the definition of conditional probability applied $n-1$ times. *Expected:* $P(X_1,\dots,X_n)=P(X_n\mid X_{<n})P(X_{<n})=\dots=\prod_{i=1}^n P(X_i\mid X_{<i})$. Mark the application of $P(A,B)=P(A\mid B)P(B)$ at each step.
20. **(advanced, interpretation, `autoregressive-lm-bridge`)** A language model with vocab size 50,257 outputs a uniform distribution over all tokens for every position. On a held-out string of 100 tokens, what is its NLL? Convert to per-token NLL. Briefly explain why this number is the worst NLL a "well-typed" model can have on this corpus. *Expected:* NLL $=100\cdot\log(50257)\approx 100\cdot 10.825\approx 1082.5$ nats. Per-token NLL $\approx 10.825$ nats. Uniform is maximum-entropy → maximum NLL given valid normalization; any other distribution would lower NLL on *some* string but uniform is the "I learned nothing" baseline. (Pay-back to m9: this is $\log|\mathcal V|$, the max cross-entropy.)

---

## 8. Endgame Callback — Refined Candidates

The starter callback is good but a hair long and front-loads three concepts at once. Three sharper candidates, each opinionated about *what to emphasize*:

**Candidate A (recommended) — Karpathy-flavored, action-oriented:**
> "Training a language model is one thing: you tune $\theta$ so $\sum_i\log P(x_i\mid\theta)$ is as large as possible. Sampling from it is the other thing: you draw from a categorical over the vocabulary, one token at a time. Every 'cross-entropy' you'll see in m9, every 'temperature' in m16, and every 'perplexity' in m18 is a sentence in the dialect this module just taught you."

**Candidate B — 3B1B-flavored, picture-first:**
> "Imagine a probability bar chart over 50,000 tokens. Training is the act of shaping that chart so the data you have is most likely; sampling is the act of dropping a uniform random number onto the chart's CDF. Those are the two verbs of every language model. Everything else — attention, layers, optimizers — is just how we *build* and *update* that bar chart."

**Candidate C — minimalist, one-sentence:**
> "An LM is a function from context to a probability vector; training is MLE on that vector; sampling is inverse-CDF on that vector — and you now have all three pieces."

**Recommendation:** Use Candidate A. It names the three downstream modules by ID, which gives the learner a Gantt-chart of what's coming. Candidate B is prettier but vaguer. Candidate C is too terse for the module wrap-up but works as a section divider in the m9 opener.

---

## 9. Sources (Licensing-Aware)

### 9.1 Andrej Karpathy — makemore Part 1 + nanoGPT code
- **Author:** Andrej Karpathy
- **URLs:** https://www.youtube.com/watch?v=PaCmpygFfXo (video); https://github.com/karpathy/makemore; https://github.com/karpathy/nanoGPT; https://github.com/karpathy/ng-video-lecture
- **Medium:** video + code repositories
- **License:** Video — All Rights Reserved (YouTube). Code repos (makemore, nanoGPT, nanochat, ng-video-lecture) — **MIT License** (verified at https://github.com/karpathy/nanoGPT/blob/master/LICENSE).
- **Tag:** Video [REFERENCE-ONLY]; Code [ADAPT].
- **Use for:** The bigram-as-MLE pedagogy and the `torch.multinomial` (= inverse-CDF) sampling pattern are the spine of Lesson 8.4 and Lesson 8.5. Pseudocode in §2.8 mirrors the makemore sampling loop. Cite the video for inspiration only; reimplement code in JS/Svelte under our own copyright with MIT attribution for any direct port.

### 9.2 3Blue1Brown — Bayes' Theorem and probability videos
- **Author:** Grant Sanderson
- **URLs:** https://www.youtube.com/watch?v=HZGCoVF3YvM ("Bayes theorem, the geometry of changing beliefs"); https://www.youtube.com/watch?v=8idr1WZ1A7Q ("Binomial distributions | Probabilities of probabilities, part 1"); https://www.youtube.com/watch?v=ZA4JkHKZM50 ("Why π is in the formula for a Gaussian")
- **Medium:** video
- **License:** All Rights Reserved (3Blue1Brown YouTube channel).
- **Tag:** [REFERENCE-ONLY].
- **Use for:** Pedagogy reference. The natural-frequency framing for Bayes (the disease-test problem, §2.2 and `bayesGrid`) is canonical 3B1B. Watch, internalize, then build our own widget that does the same job with our own assets.

### 9.3 Mathematics for Machine Learning — Chapter 6
- **Authors:** Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong
- **URL:** https://mml-book.github.io/ (free PDF) — Chapter 6 Probability and Distributions
- **Medium:** textbook (PDF + Cambridge UP print)
- **License:** Verified at mml-book.com: "PDF version is free to view and download for personal use only. **Not for re-distribution, re-sale, or use in derivative works.**" Copyright 2020 by the authors / Cambridge UP. **Not** open-licensed.
- **Tag:** [REFERENCE-ONLY].
- **Use for:** Notation discipline (the book's $p_X$ vs $f_X$ convention is exactly what we want adult learners to internalize). Worked-example structure for joint/marginal/conditional in §6.2. Do not lift figures or text.

### 9.4 Seeing Theory (Brown / Daniel Kunin)
- **Author:** Daniel Kunin, Brown University
- **URL:** https://seeing-theory.brown.edu/
- **Medium:** interactive widget site (D3)
- **License:** No formal CC license. The GitHub README states: "Feel free to use Seeing Theory for educational purposes, but we ask that you do not use the visualizations for commercial use." Project is no longer maintained.
- **Tag:** [REFERENCE-ONLY] (commercial use explicitly disallowed; this is a paid product).
- **Use for:** Visualization inspiration for `cdfSampler`, `jointMarginalScrub`, and `gaussianForge`. Their "Probability Distributions" and "Compound Probability" panels are the gold standard. Do not embed or copy code; build clean-room equivalents in Svelte 5 + Mafs.

### 9.5 Setosa — Conditional Probability
- **Author:** Victor Powell / Setosa
- **URL:** https://setosa.io/conditional/
- **Medium:** interactive widget
- **License:** No explicit open license on the article; Setosa is a commercial entity.
- **Tag:** [REFERENCE-ONLY].
- **Use for:** The "balls falling on shelves" metaphor for $P(B\mid A)$ is the cleanest visual gloss of conditional-as-renormalization on the web. Use as design reference for `bayesGrid` and `jointMarginalScrub`.

### 9.6 OpenStax Introductory Statistics (1st edition, 2018)
- **Authors:** Barbara Illowsky, Susan Dean, et al.
- **URL:** https://openstax.org/details/books/introductory-statistics (1st ed.)
- **Medium:** textbook (web + PDF)
- **License:** **CC BY 4.0** for the 1st edition (verified at https://openstax.org/books/introductory-statistics/pages/preface). **CRITICAL: the 2e (2023) is CC BY-NC-SA 4.0 — do not use 2e for adaptation; only the 1st edition is BY.**
- **Tag:** [ADAPT] (1st edition only).
- **Use for:** Problem-bank examples, especially basic-probability problems in Chapters 3–6, can be adapted with attribution. Use as a source of additional novice/intermediate problems if §7 needs expansion. Verify edition before lifting any text.

### 9.7 OpenStax Introductory Business Statistics (1st and 2nd ed)
- **Authors:** Alexander Holmes, Barbara Illowsky, Susan Dean
- **URL:** https://openstax.org/details/books/introductory-business-statistics-2e
- **Medium:** textbook
- **License:** **CC BY 4.0** for both editions (verified on openstax.org).
- **Tag:** [ADAPT].
- **Use for:** Lottery / expected-value examples (§2.5) adapt cleanly from the discrete-RV chapter. Bernoulli/binomial worked problems are also strong here.

### 9.8 Distill.pub — "A Visual Exploration of Gaussian Processes" (and related)
- **Authors:** Görtler, Kehlbeck, Deussen et al.
- **URL:** https://distill.pub/2019/visual-exploration-gaussian-processes/
- **Medium:** interactive web article
- **License:** **CC BY 4.0** (Distill's blanket policy, verified on the article page and at https://distill.pub/journal/). All Distill articles are CC BY 4.0.
- **Tag:** [ADAPT].
- **Use for:** Visualization inspiration and the Gaussian-as-affine-transform framing in `gaussianForge`. Code patterns can be adapted with attribution.

### 9.9 Nicky Case — Explorable Explanations
- **Author:** Nicky Case
- **URLs:** https://ncase.me/ and https://explorabl.es/
- **Medium:** interactive web essays
- **License:** Most projects released under CC0 / public-domain dedication per ncase.me — **verify per project before adapting**.
- **Tag:** [ADAPT] with per-project verification.
- **Use for:** Tone reference for the "two-handed interactive" feel of widgets (you're playing, not reading). For any specific code or art assets, check the GitHub repo for that project before lifting.

### 9.10 Stanford CS229 — Probability Theory Review
- **Authors:** Arian Maleki, Tom Do
- **URL:** https://cs229.stanford.edu/section/cs229-prob.pdf
- **Medium:** lecture-notes PDF
- **License:** No explicit open license stated on the document; default Stanford copyright applies.
- **Tag:** [REFERENCE-ONLY].
- **Use for:** Reference for the level of formality appropriate to ML-bound adult learners. The notes' framing of "you do not need measure theory; you need to be fluent with what's here" is exactly the right altitude for this module. Mine for problem structure; do not lift verbatim.

### Sources explicitly considered and **rejected** for incorporation
- **MIT OCW 18.05 / 6.041** — CC BY-NC-SA → reference-only.
- **Khan Academy probability** — CC BY-NC-SA → reference-only.
- **Wikipedia / Wikibooks** — CC BY-SA → reference-only.
- **OpenIntro Statistics** — CC BY-SA 3.0 → reference-only (Share-alike incompatible with a paid commercial product).
- **OpenStax Introductory Statistics 2e** — CC BY-NC-SA → reference-only (only the 1st edition is CC BY).
- **StatLect (Marco Taboga)** — "freely accessible but copyrighted," no CC license → reference-only.

---

## 10. Pedagogical Traps

### Trap 1: Hiding behind formalism (measure-theory creep)
**The trap:** Instructor reflex says "we should at least mention $\sigma$-algebras / Borel sets / measurable functions for honesty." Adult technical learners — especially the math-curious ones — feel obligated to follow, lose three days, and never reach MLE.
**Why it happens:** The instructor knows measure theory and feels uneasy "lying." The student feels every formal-looking line is essential.
**Mitigation:** Make one explicit "what we are *not* covering" callout: "Probability has a deeper foundation called measure theory; for everything in this course it gives the same answers as the rules we're using. We mention it so you know the rabbit hole exists, and we will not descend it." Then never look back. Put this aside in 8.1 step 3.

### Trap 2: Treating "density" and "probability" as the same word for ≤ 60 minutes too long
**The trap:** The pdf is introduced as "the continuous version of the pmf" without ever saying $f(x)$ can exceed 1. Two lessons later, when the Gaussian peak prints as 0.4 and a uniform on $[0,0.1]$ prints as 10, the learner thinks something is broken.
**Why it happens:** Pmf and pdf are visually almost identical. Textbooks reinforce the parallel.
**Mitigation:** In 8.1 step 6, deliberately drag the `gaussianForge` $\sigma$ to a tiny value and let the learner watch the peak grow past 1. Force the question: "is this a bug?" The lesson lands as a moment of surprise, not a definition.

### Trap 3: Bayes' theorem taught with vocabulary first instead of with population grids
**The trap:** "Posterior, likelihood, prior, evidence." Four unfamiliar words attached to one formula. Learner memorizes the formula, forgets it in a week, and never trusts it.
**Why it happens:** It's how every textbook does it because it's how every textbook before it did it.
**Mitigation:** In 8.2, start with the natural-frequency grid (`bayesGrid`) and do the entire disease-test problem *before* anyone hears the word "posterior." Introduce the vocabulary as labels on objects the learner is already pointing at. This is exactly the 3B1B / Gigerenzer framing — it is well-supported in cognitive-science research.

### Trap 4: Hand-waving the i.i.d. assumption in MLE
**The trap:** The instructor writes $L(\theta)=\prod_i P(x_i\mid\theta)$ and moves on. The learner has just silently accepted the strongest assumption in all of statistics.
**Why it happens:** It looks like a definition, not an assumption.
**Mitigation:** In 8.4 step 2, *name* the i.i.d. assumption out loud — "this product form is true *because and only because* we are treating data points as independent draws from the same distribution." Then in 8.4 step 11, foreshadow: "next module the LM will still use a product of conditionals, but the conditionals will not be independent — that's the chain rule, not i.i.d., and it's the reason 'context' exists at all." This one move buys five future modules' worth of clarity.

### Trap 5: Letting "sampling" feel like a black box
**The trap:** `torch.multinomial` and `np.random.choice` exist; learner uses them; learner has no idea what they do. Two modules later, "temperature" is a magic incantation rather than "$\mathbf{p}^{1/T}/Z$."
**Why it happens:** Library APIs are too good. The cost of running them is too low.
**Mitigation:** 8.5 *literally writes the function* (eight lines), debugs it interactively in `cdfSampler`, then declares it the function we will paste into the capstone. The learner ships a real sampler before m9 even starts. No black boxes.

### Trap 6: Spending equal time on confidence intervals and hypothesis testing
**The trap:** Statistics textbooks devote 40% of pages to inference (CIs, p-values, t-tests, ANOVA). It is irrelevant to training transformers. Including it because "a probability module ought to cover it" wastes a third of the module.
**Why it happens:** Conventional curricula assume the goal is a stats class. Ours isn't.
**Mitigation:** State the policy explicitly in 8.3 step 13: "Frequentist inference (confidence intervals, p-values, hypothesis tests) is a beautiful and important subject. It is *not* on the critical path to training a language model. We mention LLN and CLT because they calibrate intuition about sample-mean noise; we skip the rest. If you want it, read OpenIntro Statistics Chapter 5." Free the time for MLE, the chain rule, and sampling — the things that *do* show up in the transformer.
