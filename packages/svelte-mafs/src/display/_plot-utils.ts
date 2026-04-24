import type { Sample } from "../sampling.js";

/**
 * Build an SVG `d` string from 2-D samples.
 *
 * Uses explicit `M x y L x y` form (no H/V shortcuts) so tests can count
 * segments by regex and so NaN-marker points break the stroke cleanly:
 * a non-finite coordinate flushes the current sub-path and forces the next
 * finite point to emit a fresh `M`.
 */
export const buildPath = (samples: readonly Sample[]): string => {
  const parts: string[] = [];
  let pendingMove = true;
  for (const [x, y] of samples) {
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      pendingMove = true;
      continue;
    }
    parts.push(`${pendingMove ? "M" : "L"} ${x} ${y}`);
    pendingMove = false;
  }
  return parts.join(" ");
};

/**
 * A tolerance that trades acceptable error (in user-y units) for fewer
 * samples. 500 subdivisions worth of resolution across the visible y-span
 * is imperceptible at any reasonable Mafs size.
 */
export const defaultTolerance = (ySpan: number): number => ySpan / 500;

export type StrokeStyle = "solid" | "dashed" | "dotted";

/** Map style token to an SVG stroke-dasharray. Null → no dasharray attr. */
export const dashArrayFor = (style: StrokeStyle): string | null => {
  if (style === "dashed") return "6 6";
  if (style === "dotted") return "2 4";
  return null;
};

/**
 * Treat values beyond this magnitude as asymptotic. Matches the scalar
 * sampler in ../sampling.ts so both sampling paths break on the same
 * numeric cliff.
 */
const DIVERGENCE_THRESHOLD = 1e10;

const isFinitePoint = (p: readonly [number, number]): boolean =>
  Number.isFinite(p[0]) &&
  Number.isFinite(p[1]) &&
  Math.abs(p[0]) < DIVERGENCE_THRESHOLD &&
  Math.abs(p[1]) < DIVERGENCE_THRESHOLD;

export interface ParametricSampleOptions {
  readonly domain: readonly [number, number];
  /** Chord-distance tolerance in output (x, y) units before subdividing further. */
  readonly tolerance: number;
  readonly maxDepth: number;
  readonly minDepth?: number;
}

/**
 * Adaptive midpoint-subdivision sampler for 2-D parametric curves.
 *
 * Mirrors the scalar sampler's flatness strategy, but measures error as
 * chord distance in (x, y) space so curves with both coordinates changing
 * simultaneously (circles, Lissajous figures) sample densely where needed.
 * NaN or divergent outputs produce a `[NaN, NaN]` marker so buildPath
 * breaks the stroke.
 */
export const sampleParametric = (
  f: (t: number) => readonly [number, number],
  opts: ParametricSampleOptions,
): Array<readonly [number, number]> => {
  let [tMin, tMax] = opts.domain;
  if (tMin > tMax) [tMin, tMax] = [tMax, tMin];
  if (tMin === tMax) return [f(tMin)];

  const { tolerance, maxDepth } = opts;
  const minDepth = opts.minDepth ?? 4;

  const subdivide = (
    tL: number,
    pL: readonly [number, number],
    tR: number,
    pR: readonly [number, number],
    depth: number,
  ): Array<readonly [number, number]> => {
    const tm = (tL + tR) / 2;
    const pm = f(tm);

    if (!isFinitePoint(pm)) {
      return [pL, [Number.NaN, Number.NaN], pR];
    }

    if (depth < minDepth) {
      const left = subdivide(tL, pL, tm, pm, depth + 1);
      const right = subdivide(tm, pm, tR, pR, depth + 1);
      return [...left, ...right.slice(1)];
    }

    const midX = (pL[0] + pR[0]) / 2;
    const midY = (pL[1] + pR[1]) / 2;
    const flat =
      isFinitePoint(pL) &&
      isFinitePoint(pR) &&
      Math.hypot(pm[0] - midX, pm[1] - midY) < tolerance;

    if (flat || depth >= maxDepth) {
      return [pL, pm, pR];
    }

    const left = subdivide(tL, pL, tm, pm, depth + 1);
    const right = subdivide(tm, pm, tR, pR, depth + 1);
    return [...left, ...right.slice(1)];
  };

  return subdivide(tMin, f(tMin), tMax, f(tMax), 0);
};
