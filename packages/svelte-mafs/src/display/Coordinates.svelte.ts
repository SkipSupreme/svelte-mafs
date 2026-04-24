import Cartesian from "./coordinates-cartesian.svelte";

/**
 * Consumer API:
 *   import { Coordinates } from "svelte-mafs";
 *   <Coordinates.Cartesian />
 *
 * Reserved shape: Polar/other coordinate systems land as additional keys
 * in a future version. Kept as a namespace object (rather than a single
 * component) so the addition is non-breaking.
 */
export const Coordinates = {
  Cartesian,
} as const;

export default Coordinates;
