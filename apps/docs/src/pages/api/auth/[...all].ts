import type { APIRoute } from 'astro';
import { createAuth } from '../../../server/auth';
import { getEnv } from '../../../server/env';

export const prerender = false;

const handler: APIRoute = async ({ request }) => {
  const env = getEnv();
  const auth = createAuth({
    DB: env.DB,
    BETTER_AUTH_SECRET: env.BETTER_AUTH_SECRET,
    PUBLIC_SITE_URL: env.PUBLIC_SITE_URL ?? new URL(request.url).origin,
  });
  return auth.handler(request);
};

export const GET = handler;
export const POST = handler;
export const PUT = handler;
export const DELETE = handler;
export const PATCH = handler;
export const OPTIONS = handler;
export const HEAD = handler;
