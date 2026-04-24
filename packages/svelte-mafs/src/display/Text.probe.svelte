<script lang="ts">
  // Test-only wrapper: mounts <Text> inside a <Mafs> so the coord context is
  // available. Mirrors the Mafs.probe pattern from Stream 2.
  import Mafs from "../view/Mafs.svelte";
  import Text from "./Text.svelte";

  type Props = {
    x: number;
    y: number;
    latex: string;
    size?: number;
    color?: string;
  };

  const props: Props = $props();
</script>

<Mafs width={400} height={400} viewBox={{ x: [-5, 5], y: [-5, 5] }}>
  <Text {...props} />
</Mafs>
