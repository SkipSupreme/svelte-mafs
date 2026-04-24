import { getContext, setContext } from "svelte";
import type { NormalizedViewBox } from "./coordinate-context.js";

// Nested panes let a sub-tree of a <Mafs> diagram render against a different
// visible range than the root. The full pane feature (clipping, sub-svg, etc.)
// lands in a later stream; this module is the context plumbing.
export interface PaneContext {
  readonly viewBox: NormalizedViewBox;
}

const KEY = Symbol("mafs.pane");

export const setPaneContext = (ctx: PaneContext): PaneContext =>
  setContext(KEY, ctx);

export const getPaneContext = (): PaneContext | undefined =>
  getContext<PaneContext | undefined>(KEY);
