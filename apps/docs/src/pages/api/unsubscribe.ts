import type { APIRoute } from 'astro';
import { eq } from 'drizzle-orm';
import { getDb } from '../../server/db';
import { userProfile } from '../../server/schema';
import { verifyUnsubscribeToken } from '../../server/email';
import { getEnv } from '../../server/env';

export const prerender = false;

const PAGE = (status: 'ok' | 'bad') => `<!doctype html>
<html lang="en"><head>
<meta charset="utf-8" />
<title>Unsubscribed</title>
<style>
  html { font-family: ui-sans-serif, system-ui, sans-serif; }
  main { max-width: 480px; margin: 4rem auto; padding: 2rem; text-align: center; }
  h1 { font-size: 1.5rem; margin-bottom: 0.5rem; }
  p { color: #666; }
</style>
</head><body><main>
${status === 'ok'
  ? '<h1>You\'re unsubscribed.</h1><p>You won\'t receive any more module-launch emails from Tinker. Sign-in and account emails will still go through. <a href="/me">Visit your account</a> to opt back in any time.</p>'
  : '<h1>This link is no longer valid.</h1><p>If you\'d still like to unsubscribe, <a href="/me">visit your account page</a>.</p>'}
</main></body></html>`;

export const GET: APIRoute = async ({ url, locals }) => {
  const env = getEnv();
  const token = url.searchParams.get('token');
  if (!token || !env.UNSUBSCRIBE_HMAC_SECRET) {
    return new Response(PAGE('bad'), { status: 400, headers: { 'content-type': 'text/html; charset=utf-8' } });
  }
  const userId = await verifyUnsubscribeToken(token, env.UNSUBSCRIBE_HMAC_SECRET);
  if (!userId) {
    return new Response(PAGE('bad'), { status: 400, headers: { 'content-type': 'text/html; charset=utf-8' } });
  }
  const db = getDb(env.DB);
  await db
    .update(userProfile)
    .set({ marketingOptIn: false })
    .where(eq(userProfile.userId, userId));
  return new Response(PAGE('ok'), { status: 200, headers: { 'content-type': 'text/html; charset=utf-8' } });
};

// One-click unsubscribe POST per RFC 8058
export const POST = GET;
