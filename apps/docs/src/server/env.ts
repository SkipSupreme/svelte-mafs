import { env as cfEnv } from 'cloudflare:workers';

/**
 * Typed Cloudflare Worker env. Astro v6 removed the
 * Astro-locals-runtime-env access pattern; the new pattern is to import
 * the binding from `cloudflare:workers` and cast to our typed `Env`.
 */
export function getEnv(): Env {
  return cfEnv as unknown as Env;
}
