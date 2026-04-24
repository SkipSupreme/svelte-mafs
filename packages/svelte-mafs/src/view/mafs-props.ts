import type { Snippet } from "svelte";
import type { UserViewBox } from "../context/coordinate-context.js";

export type MafsZoomOption = boolean | { readonly min: number; readonly max: number };

export interface MafsProps {
  width: number | "auto";
  height: number | "auto";
  viewBox: UserViewBox;
  preserveAspectRatio?: boolean;
  // Accepted but inert in Stream 2. Gesture actions land in Stream 3; the
  // wiring that drives viewBox state lands in Stream 6.
  pan?: boolean;
  zoom?: MafsZoomOption;
  children?: Snippet;
}
