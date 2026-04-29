import type { APIRoute } from 'astro';
import { createAuth } from '../../../server/auth';
import { getDb } from '../../../server/db';
import { getEnv } from '../../../server/env';
import { checkRateLimit } from '../../../server/ratelimit';
import { ResendError } from '../../../server/email';

export const prerender = false;

const handler: APIRoute = async ({ request }) => {
  const env = getEnv();
  const url = new URL(request.url);
  const auth = createAuth({
    DB: env.DB,
    BETTER_AUTH_SECRET: env.BETTER_AUTH_SECRET,
    GOOGLE_CLIENT_ID: env.GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET: env.GOOGLE_CLIENT_SECRET,
    GITHUB_CLIENT_ID: env.GITHUB_CLIENT_ID,
    GITHUB_CLIENT_SECRET: env.GITHUB_CLIENT_SECRET,
    RESEND_API_KEY: env.RESEND_API_KEY,
    PUBLIC_SITE_URL: env.PUBLIC_SITE_URL ?? url.origin,
  });

  // Magic-link send is the priciest endpoint (calls Resend, sends email).
  // Cap by client IP so the form can't be turned into an email cannon.
  if (request.method === 'POST' && url.pathname.endsWith('/sign-in/magic-link')) {
    const ip = clientIp(request);
    const db = getDb(env.DB);
    const rl = await checkRateLimit(db, `auth:magic-link:${ip}`, { limit: 5, windowMs: 60_000 });
    if (!rl.allowed) {
      return new Response(
        JSON.stringify({ error: 'Too many sign-in requests, try again in a minute.' }),
        {
          status: 429,
          headers: {
            'content-type': 'application/json',
            'retry-after': String(Math.ceil(rl.retryAfterMs / 1000)),
          },
        },
      );
    }
  }

  try {
    return await auth.handler(request);
  } catch (err) {
    // Resend transport failures (unverified domain, revoked key, etc.)
    // bubble up from sendMagicLinkEmail. Map to a clean 503 + logged
    // detail so users see "service unavailable" instead of a Cloudflare
    // generic 500 with no body.
    if (err instanceof ResendError) {
      console.error('[auth] resend failed', err.status, err.body);
      return new Response(
        JSON.stringify({
          error: "Sign-in email service is temporarily unavailable. We've been notified.",
        }),
        { status: 503, headers: { 'content-type': 'application/json' } },
      );
    }
    throw err;
  }
};

function clientIp(request: Request): string {
  return (
    request.headers.get('cf-connecting-ip') ??
    request.headers.get('x-forwarded-for')?.split(',')[0]?.trim() ??
    'unknown'
  );
}

export const GET = handler;
export const POST = handler;
export const PUT = handler;
export const DELETE = handler;
export const PATCH = handler;
export const OPTIONS = handler;
export const HEAD = handler;
