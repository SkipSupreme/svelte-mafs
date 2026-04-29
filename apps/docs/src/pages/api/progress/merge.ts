import type { APIRoute } from 'astro';
import { z } from 'zod';
import { requireSession, requireCsrf, jsonError, jsonOk } from '../../../server/middleware';
import { checkRateLimit } from '../../../server/ratelimit';
import { mergeAnonProgress } from '../../../server/progress';
import { getEnv } from '../../../server/env';

export const prerender = false;

const EntrySchema = z.object({
  lesson_slug: z.string().min(1).max(200),
  course_slug: z.string().min(0).max(100),
  module_slug: z.string().min(0).max(100),
  first_seen_at: z.string().min(1),
  last_seen_at: z.string().min(1),
  view_count: z.number().int().nonnegative().optional(),
  completed_at: z.string().nullable().optional(),
});

const Body = z.object({
  entries: z.array(EntrySchema).max(500),
});

export const POST: APIRoute = async ({ request, locals }) => {
  const env = getEnv();
  const csrf = requireCsrf(request);
  if (csrf) return csrf;

  const ctx = await requireSession(request, env);
  if ('error' in ctx) return ctx.error;

  const rl = await checkRateLimit(ctx.db, `progress:merge:${ctx.session.user.id}`, {
    limit: 5,
    windowMs: 60_000,
  });
  if (!rl.allowed) return jsonError(429, 'rate_limited', 'Too many merges');

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return jsonError(400, 'bad_json', 'Could not parse JSON body');
  }
  const parsed = Body.safeParse(body);
  if (!parsed.success) return jsonError(400, 'bad_request', 'Invalid payload');

  const result = await mergeAnonProgress(ctx.db, ctx.session.user.id, parsed.data.entries);
  return jsonOk(result);
};
