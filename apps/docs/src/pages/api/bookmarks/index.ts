import type { APIRoute } from 'astro';
import { z } from 'zod';
import { requireSession, requireCsrf, jsonError, jsonOk } from '../../../server/middleware';
import { checkRateLimit } from '../../../server/ratelimit';
import { createBookmark, listBookmarks } from '../../../server/bookmarks';
import { getEnv } from '../../../server/env';

export const prerender = false;

const Body = z.object({
  lesson_slug: z.string().min(1).max(200),
  anchor: z.string().max(200).nullable().optional(),
});

export const POST: APIRoute = async ({ request, locals }) => {
  const env = getEnv();
  const csrf = requireCsrf(request);
  if (csrf) return csrf;

  const ctx = await requireSession(request, env);
  if ('error' in ctx) return ctx.error;

  const rl = await checkRateLimit(ctx.db, `bookmark:${ctx.session.user.id}`, {
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

  const r = await createBookmark(
    ctx.db,
    ctx.session.user.id,
    parsed.data.lesson_slug,
    parsed.data.anchor ?? null,
  );
  return jsonOk(r, r.created ? 201 : 200);
};

export const GET: APIRoute = async ({ request, locals }) => {
  const env = getEnv();
  const ctx = await requireSession(request, env);
  if ('error' in ctx) return ctx.error;
  const list = await listBookmarks(ctx.db, ctx.session.user.id);
  return jsonOk({ bookmarks: list });
};
