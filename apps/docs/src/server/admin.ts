import { eq } from 'drizzle-orm';
import { user as userTbl } from './schema';
import { createAuth, getSession, type AuthEnv } from './auth';
import { getDb } from './db';

export async function requireAdmin(
  request: Request,
  env: AuthEnv,
): Promise<
  | { ok: true; userId: string; email: string; db: ReturnType<typeof getDb> }
  | { ok: false; response: Response }
> {
  const auth = createAuth(env);
  const session = await getSession(auth, request.headers);
  if (!session) {
    return { ok: false, response: notFound() };
  }
  const db = getDb(env.DB);
  const u = await db.select().from(userTbl).where(eq(userTbl.id, session.user.id)).get();
  if (!u || u.role !== 'admin') {
    return { ok: false, response: notFound() };
  }
  return { ok: true, userId: u.id, email: u.email, db };
}

/**
 * Returns a generic 404 response — we deliberately don't 403 admin routes
 * so the surface is invisible to non-admins.
 */
function notFound(): Response {
  return new Response('<!doctype html><title>404</title><h1>Not found</h1>', {
    status: 404,
    headers: { 'content-type': 'text/html; charset=utf-8' },
  });
}
