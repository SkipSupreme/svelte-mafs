# M18 engine — slice 2 verification

Two scripts live here:

- `nanogpt_reference.py` — produces `docs/research/m18-nanogpt-reference.csv`,
  the loss-trajectory oracle Slice 3's WGSL training loop must match
  (±0.1 nat envelope at every 100-iter checkpoint).
- `gpu_smoke.ts` — headless WGSL-vs-CPU equivalence runner.

## Headless WebGPU runtime choice: **Bun**

The brief asked us to pick between Bun (built-in WebGPU) and Node + `@webgpu/dawn`
(native binding). We picked **Bun** for these reasons:

1. No native build step. Node's `@webgpu/dawn` requires a Dawn compile that's
   slow to land on every dev machine.
2. The same Bun runtime will host Slice 3's CLI training run for the twin-seed
   determinism check (Slice 4) — keeping the runtime stack narrow.
3. The `EngineCore` class in `apps/docs/src/lib/m18/engine/core.ts` is
   specifically structured to accept injected kernel sources so Bun and the
   browser share the same forward path.

**Caveat (April 2026):** The Bun release on this dev box (1.3.11) ships
without `navigator.gpu`. WebGPU is on the Bun roadmap and behind a flag in
nightly builds; the smoke script detects the missing API and exits with a
clear message until then. The in-browser smoke at `/dev/m18-forward/` is the
canonical Slice 2 verification meanwhile, and the CPU-twin vitest suite
(`apps/docs/src/lib/m18/engine/cpu/twins.test.ts`) pins the math reference
the WGSL kernels must match.

## Running it

```sh
# Reference oracle (PyTorch on CPU, ~minutes).
python3 scripts/m18/nanogpt_reference.py

# Headless GPU smoke (Bun, when WebGPU lands).
bun scripts/m18/gpu_smoke.ts

# CPU-twin sanity (always works).
pnpm -F docs test
```
