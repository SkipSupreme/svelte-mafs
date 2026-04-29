import { createAuth, getSession, type AuthEnv, type SessionContext } from './auth';
import { getDb, type DB } from './db';

const CSRF_COOKIE_PATTERN = /(?:^|;\s*)__Secure-tinker\.csrf_token=([^;]+)/;
const CSRF_DEV_COOKIE_PATTERN = /(?:^|;\s*)tinker\.csrf_token=([^;]+)/;

export interface SessionedContext {
  session: SessionContext;
  db: DB;
  env: AuthEnv;
  auth: ReturnType<typeof createAuth>;
}

export type RequireSessionResult =
  | SessionedContext
  | { error: Response };

export async function requireSession(
  request: Request,
  env: AuthEnv,
): Promise<RequireSessionResult> {
  const auth = createAuth(env);
  const session = await getSession(auth, request.headers);
  if (!session) {
    return { error: jsonError(401, 'unauthorized', 'Sign in required') };
  }
  return { session, db: getDb(env.DB), env, auth };
}

export function requireCsrf(request: Request): Response | null {
  const cookieHeader = request.headers.get('cookie') ?? '';
  const match =
    cookieHeader.match(CSRF_COOKIE_PATTERN) ?? cookieHeader.match(CSRF_DEV_COOKIE_PATTERN);
  const csrfCookie = match?.[1];
  const csrfHeader = request.headers.get('x-tinker-csrf');
  if (!csrfCookie || !csrfHeader || !timingSafeEqual(csrfCookie, csrfHeader)) {
    return jsonError(403, 'forbidden', 'CSRF check failed');
  }
  return null;
}

function timingSafeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let mismatch = 0;
  for (let i = 0; i < a.length; i++) mismatch |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return mismatch === 0;
}

export function jsonError(status: number, code: string, message: string): Response {
  return new Response(JSON.stringify({ error: { code, message } }), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

export function jsonOk<T>(data: T, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}
