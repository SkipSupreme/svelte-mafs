// @ts-check
import { defineConfig } from 'astro/config';

import svelte from '@astrojs/svelte';
import cloudflare from '@astrojs/cloudflare';
import mdx from '@astrojs/mdx';

// https://astro.build/config
export default defineConfig({
  site: 'https://svelte-mafs.pages.dev',
  integrations: [svelte(), mdx()],
  adapter: cloudflare(),
  output: 'static',
  markdown: {
    shikiConfig: {
      themes: { light: 'github-light', dark: 'github-dark' },
      wrap: true,
    },
  },
});
