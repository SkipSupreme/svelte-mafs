import type { APIRoute } from 'astro';
import { z } from 'zod';
import { requireSession, requireCsrf, jsonError, jsonOk } from '../../../server/middleware';
import { checkRateLimit } from '../../../server/ratelimit';
import { upsertNote, getNote, NOTE_MAX_BYTES } from '../../../server/notes';
import { getEnv } from '../../../server/env';

export const prerender = false;

const Body = z.object({
  body: z.string().max(NOTE_MAX_BYTES),
});

export const PUT: APIRoute = async ({ request, params, locals }) => {
  const env = getEnv();
  const csrf = requireCsrf(request);
  if (csrf) return csrf;

  const ctx = await requireSession(request, env);
  if ('error' in ctx) return ctx.error;

  const lessonSlug = params.lesson;
  if (!lessonSlug || lessonSlug.length > 200) {
    return jsonError(400, 'bad_request', 'Missing or invalid lesson slug');
  }

  const rl = await checkRateLimit(ctx.db, `notes:${ctx.session.user.id}`, {
    limit: 120,
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
  if (!parsed.success) {
    if (typeof (body as { body?: unknown })?.body === 'string') {
      return jsonError(413, 'too_large', 'Note exceeds 50KB');
    }
    return jsonError(400, 'bad_request', 'Invalid payload');
  }

  const r = await upsertNote(ctx.db, ctx.session.user.id, lessonSlug, parsed.data.body);
  return jsonOk({ updatedAt: r.updatedAt.toISOString() });
};

export const GET: APIRoute = async ({ request, params, locals }) => {
  const env = getEnv();
  const ctx = await requireSession(request, env);
  if ('error' in ctx) return ctx.error;
  const lessonSlug = params.lesson;
  if (!lessonSlug) return jsonError(400, 'bad_request', 'Missing lesson slug');
  const r = await getNote(ctx.db, ctx.session.user.id, lessonSlug);
  return jsonOk({ body: r?.body ?? '', updatedAt: r?.updatedAt?.toISOString() ?? null });
};
