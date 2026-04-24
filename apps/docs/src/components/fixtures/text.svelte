<script lang="ts">
  import { Mafs, Coordinates, Text } from "svelte-mafs";

  // LaTeX as script-side strings so Svelte doesn't try to interpret `\p`
  // etc. as JS string escapes in attribute position.
  const eulersIdentity = String.raw`e^{i\pi} + 1 = 0`;
  const baselZeta = String.raw`\sum_{n=1}^{\infty} \frac{1}{n^2}`;
  // KaTeX CSS is not imported — apps/docs doesn't declare `katex` as a
  // direct dep (it comes transitively via svelte-mafs). The rendered
  // fallback HTML is still visually stable for regression purposes.
</script>

<Mafs width={360} height={360} viewBox={{ x: [-5, 5], y: [-5, 5] }}>
  <Coordinates.Cartesian />
  <Text x={2} y={3} latex={eulersIdentity} />
  <Text x={-2} y={-2} latex={baselZeta} color="#2563eb" />
</Mafs>
