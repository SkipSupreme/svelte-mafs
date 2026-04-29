import type { APIRoute } from 'astro';
import { requireSession, requireCsrf, jsonError, jsonOk } from '../../../server/middleware';
import { deleteBookmark } from '../../../server/bookmarks';
import { getEnv } from '../../../server/env';

export const prerender = false;

export const DELETE: APIRoute = async ({ request, params, locals }) => {
  const env = getEnv();
  const csrf = requireCsrf(request);
  if (csrf) return csrf;

  const ctx = await requireSession(request, env);
  if ('error' in ctx) return ctx.error;

  const id = params.id;
  if (!id) return jsonError(400, 'bad_request', 'Missing id');

  const r = await deleteBookmark(ctx.db, ctx.session.user.id, id);
  if (!r.deleted) return jsonError(404, 'not_found', 'Bookmark not found');
  return jsonOk({ deleted: true });
};
