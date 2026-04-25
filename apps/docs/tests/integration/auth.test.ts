import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { betterAuth } from 'better-auth';
import { drizzleAdapter } from 'better-auth/adapters/drizzle';
import { magicLink } from 'better-auth/plugins';
import { makeTestDb, type TestDb } from '../support/d1';
import { session as sessionTbl, user as userTbl } from '../../src/server/schema';

interface Sent {
  email: string;
  url: string;
}

let db: TestDb;
let sent: Sent[];
let auth: ReturnType<typeof betterAuth>;

function newAuth(database: ReturnType<typeof drizzleAdapter>) {
  return betterAuth({
    database,
    secret: 'a'.repeat(64),
    baseURL: 'http://localhost',
    trustedOrigins: ['http://localhost'],
    emailAndPassword: { enabled: false },
    plugins: [
      magicLink({
        expiresIn: 60 * 15,
        sendMagicLink: async ({ email, url }) => {
          sent.push({ email, url });
        },
      }),
    ],
    advanced: {
      cookiePrefix: 'tinker',
      useSecureCookies: false,
    },
    user: {
      additionalFields: {
        role: { type: 'string', defaultValue: 'user', input: false },
      },
    },
  });
}

beforeEach(() => {
  db = makeTestDb();
  sent = [];
  auth = newAuth(drizzleAdapter(db.client, { provider: 'sqlite' })) as unknown as typeof auth;
});

afterEach(() => {
  db.close();
  vi.useRealTimers();
});

async function requestMagicLink(email: string) {
  return auth.handler(
    new Request('http://localhost/api/auth/sign-in/magic-link', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ email, callbackURL: '/' }),
    }),
  );
}

describe('magic link flow', () => {
  it('creates a verification row + sends a link on request', async () => {
    const res = await requestMagicLink('a@b.co');
    expect(res.status).toBeLessThan(400);
    expect(sent).toHaveLength(1);
    expect(sent[0].email).toBe('a@b.co');
    expect(sent[0].url).toMatch(/token=/);
  });

  it('verifies + creates session + creates user on click', async () => {
    await requestMagicLink('a@b.co');
    const verifyRes = await auth.handler(new Request(sent[0].url));
    expect(verifyRes.status).toBeLessThan(400);

    const users = await db.client.select().from(userTbl).all();
    expect(users).toHaveLength(1);
    expect(users[0].email).toBe('a@b.co');

    const sessions = await db.client.select().from(sessionTbl).all();
    expect(sessions.length).toBeGreaterThan(0);
    expect(sessions[0].userId).toBe(users[0].id);
  });

  it('rejects a re-used token (no second session row, redirected with error)', async () => {
    await requestMagicLink('a@b.co');
    const url = sent[0].url;
    await auth.handler(new Request(url));
    const sessionsAfterFirst = await db.client.select().from(sessionTbl).all();
    expect(sessionsAfterFirst).toHaveLength(1);

    const second = await auth.handler(new Request(url));
    // Better Auth redirects with ?error=... rather than returning 4xx;
    // assert no NEW session was created.
    const location = second.headers.get('Location') ?? '';
    const sessionsAfterSecond = await db.client.select().from(sessionTbl).all();
    expect(sessionsAfterSecond).toHaveLength(1);
    expect(location).toMatch(/error/i);
  });

  it('rejects an expired token (no session created, redirect carries error)', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-01-01T12:00:00Z'));
    await requestMagicLink('a@b.co');
    const url = sent[0].url;
    vi.setSystemTime(new Date('2026-01-01T12:16:00Z')); // 16 minutes later
    const res = await auth.handler(new Request(url));
    const location = res.headers.get('Location') ?? '';
    const sessions = await db.client.select().from(sessionTbl).all();
    expect(sessions).toHaveLength(0);
    expect(location).toMatch(/error/i);
  });

  it('returns same user when same email signs in twice', async () => {
    await requestMagicLink('a@b.co');
    await auth.handler(new Request(sent[0].url));
    const usersAfterFirst = await db.client.select().from(userTbl).all();
    expect(usersAfterFirst).toHaveLength(1);
    const firstUserId = usersAfterFirst[0].id;

    sent = [];
    await requestMagicLink('a@b.co');
    await auth.handler(new Request(sent[0].url));
    const usersAfterSecond = await db.client.select().from(userTbl).all();
    expect(usersAfterSecond).toHaveLength(1);
    expect(usersAfterSecond[0].id).toBe(firstUserId);
  });
});
