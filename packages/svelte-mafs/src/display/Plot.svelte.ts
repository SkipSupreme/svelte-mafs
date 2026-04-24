import OfX from "./plot-of-x.svelte";
import OfY from "./plot-of-y.svelte";
import Parametric from "./plot-parametric.svelte";
import Inequality from "./plot-inequality.svelte";
import VectorField from "./plot-vector-field.svelte";

/**
 * Consumer API:
 *   import { Plot } from "svelte-mafs";
 *   <Plot.OfX y={(x) => Math.sin(x)} />
 *
 * Svelte 5 resolves dotted component names as JS property access, so any
 * object whose values are component constructors works as a namespace.
 */
export const Plot = {
  OfX,
  OfY,
  Parametric,
  Inequality,
  VectorField,
} as const;

export default Plot;
