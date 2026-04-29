import type { APIRoute } from 'astro';
import { z } from 'zod';
import { requireSession, requireCsrf, jsonError } from '../../../server/middleware';
import { checkRateLimit } from '../../../server/ratelimit';
import { recordCompletion } from '../../../server/progress';
import { getEnv } from '../../../server/env';

export const prerender = false;

const Body = z.object({
  lesson_slug: z.string().min(1).max(200),
});

export const POST: APIRoute = async ({ request, locals }) => {
  const env = getEnv();
  const csrf = requireCsrf(request);
  if (csrf) return csrf;

  const ctx = await requireSession(request, env);
  if ('error' in ctx) return ctx.error;

  const rl = await checkRateLimit(ctx.db, `progress:complete:${ctx.session.user.id}`, {
    limit: 60,
    windowMs: 60_000,
  });
  if (!rl.allowed) return jsonError(429, 'rate_limited', 'Too many requests');

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return jsonError(400, 'bad_json', 'Could not parse JSON body');
  }
  const parsed = Body.safeParse(body);
  if (!parsed.success) return jsonError(400, 'bad_request', 'Invalid payload');

  await recordCompletion(ctx.db, ctx.session.user.id, parsed.data.lesson_slug);

  return new Response(null, { status: 204 });
};
