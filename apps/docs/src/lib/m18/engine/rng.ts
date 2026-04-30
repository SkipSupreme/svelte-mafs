// sfc32 PRNG seeded by cyrb128. Adapted from
// https://github.com/bryc/code/blob/master/jshash/PRNGs.md (public domain).
// Slice 3 uses this for weight init + train batch indices so a convergence
// run is reproducible within the slice. Slice 4 ships the full determinism
// surface (dropout masks, sampler, twin-seed test).

export function cyrb128(str: string): [number, number, number, number] {
  let h1 = 1779033703, h2 = 3144134277, h3 = 1013904242, h4 = 2773480762;
  for (let i = 0, k; i < str.length; i++) {
    k = str.charCodeAt(i);
    h1 = h2 ^ Math.imul(h1 ^ k, 597399067);
    h2 = h3 ^ Math.imul(h2 ^ k, 2869860233);
    h3 = h4 ^ Math.imul(h3 ^ k, 951274213);
    h4 = h1 ^ Math.imul(h4 ^ k, 2716044179);
  }
  h1 = Math.imul(h3 ^ (h1 >>> 18), 597399067);
  h2 = Math.imul(h4 ^ (h2 >>> 22), 2869860233);
  h3 = Math.imul(h1 ^ (h3 >>> 17), 951274213);
  h4 = Math.imul(h2 ^ (h4 >>> 19), 2716044179);
  return [
    (h1 ^ h2 ^ h3 ^ h4) >>> 0,
    (h2 ^ h1) >>> 0,
    (h3 ^ h1) >>> 0,
    (h4 ^ h1) >>> 0,
  ];
}

export function sfc32(a: number, b: number, c: number, d: number): () => number {
  let s0 = a >>> 0, s1 = b >>> 0, s2 = c >>> 0, s3 = d >>> 0;
  return () => {
    s0 |= 0; s1 |= 0; s2 |= 0; s3 |= 0;
    const t = (((s0 + s1) | 0) + s3) | 0;
    s3 = (s3 + 1) | 0;
    s0 = s1 ^ (s1 >>> 9);
    s1 = (s2 + (s2 << 3)) | 0;
    s2 = (s2 << 21) | (s2 >>> 11);
    s2 = (s2 + t) | 0;
    return (t >>> 0) / 4294967296;
  };
}

// Seed convenience: stringify any int → cyrb128 → sfc32.
export function seededRng(seed: number | string): () => number {
  const s = typeof seed === 'number' ? String(seed) : seed;
  const [a, b, c, d] = cyrb128(s);
  return sfc32(a, b, c, d);
}

// Box-Muller transform — converts two uniforms in (0, 1) to two
// independent N(0, 1) samples. Used by the He / std=0.02 weight initializer.
export function boxMuller(rng: () => number): [number, number] {
  let u1 = rng(); if (u1 < 1e-300) u1 = 1e-300;
  const u2 = rng();
  const r = Math.sqrt(-2 * Math.log(u1));
  const theta = 2 * Math.PI * u2;
  return [r * Math.cos(theta), r * Math.sin(theta)];
}

// Fill an array with N(0, std) samples.
export function fillNormal(out: Float32Array, std: number, rng: () => number): void {
  let i = 0;
  while (i + 1 < out.length) {
    const [a, b] = boxMuller(rng);
    out[i++] = a * std;
    out[i++] = b * std;
  }
  if (i < out.length) {
    const [a] = boxMuller(rng);
    out[i] = a * std;
  }
}
