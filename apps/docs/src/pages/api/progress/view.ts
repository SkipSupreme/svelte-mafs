import type { APIRoute } from 'astro';
import { z } from 'zod';
import { requireSession, requireCsrf, jsonError } from '../../../server/middleware';
import { checkRateLimit } from '../../../server/ratelimit';
import { recordView } from '../../../server/progress';
import { getEnv } from '../../../server/env';

export const prerender = false;

const Body = z.object({
  lesson_slug: z.string().min(1).max(200),
  course_slug: z.string().min(1).max(100),
  module_slug: z.string().min(1).max(100),
});

export const POST: APIRoute = async ({ request, locals }) => {
  const env = getEnv();
  const csrf = requireCsrf(request);
  if (csrf) return csrf;

  const ctx = await requireSession(request, env);
  if ('error' in ctx) return ctx.error;

  const rl = await checkRateLimit(ctx.db, `progress:view:${ctx.session.user.id}`, {
    limit: 120,
    windowMs: 60_000,
  });
  if (!rl.allowed) {
    return new Response(JSON.stringify({ error: { code: 'rate_limited', message: 'Too many requests' } }), {
      status: 429,
      headers: { 'content-type': 'application/json', 'retry-after': String(Math.ceil(rl.retryAfterMs / 1000)) },
    });
  }

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return jsonError(400, 'bad_json', 'Could not parse JSON body');
  }
  const parsed = Body.safeParse(body);
  if (!parsed.success) return jsonError(400, 'bad_request', 'Invalid payload');

  await recordView(ctx.db, ctx.session.user.id, {
    courseSlug: parsed.data.course_slug,
    moduleSlug: parsed.data.module_slug,
    lessonSlug: parsed.data.lesson_slug,
  });

  return new Response(null, { status: 204 });
};
