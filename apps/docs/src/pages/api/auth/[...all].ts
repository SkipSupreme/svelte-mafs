import type { APIRoute } from 'astro';
import { createAuth } from '../../../server/auth';
import { getEnv } from '../../../server/env';

export const prerender = false;

const handler: APIRoute = async ({ request, locals }) => {
  const env = getEnv();
  const auth = createAuth({
    DB: env.DB,
    BETTER_AUTH_SECRET: env.BETTER_AUTH_SECRET,
    GOOGLE_CLIENT_ID: env.GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET: env.GOOGLE_CLIENT_SECRET,
    GITHUB_CLIENT_ID: env.GITHUB_CLIENT_ID,
    GITHUB_CLIENT_SECRET: env.GITHUB_CLIENT_SECRET,
    RESEND_API_KEY: env.RESEND_API_KEY,
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
