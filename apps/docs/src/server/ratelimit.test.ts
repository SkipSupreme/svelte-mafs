import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { checkRateLimit } from './ratelimit';
import { makeTestDb, type TestDb } from '../../tests/support/d1';

let db: TestDb;
beforeEach(() => {
  db = makeTestDb();
});
afterEach(() => {
  db.close();
});

describe('checkRateLimit', () => {
  it('allows up to limit then blocks', async () => {
    for (let i = 0; i < 3; i++) {
      const r = await checkRateLimit(db.client, 'k', { limit: 3, windowMs: 60_000 });
      expect(r.allowed).toBe(true);
    }
    const r = await checkRateLimit(db.client, 'k', { limit: 3, windowMs: 60_000 });
    expect(r.allowed).toBe(false);
    expect(r.retryAfterMs).toBeGreaterThan(0);
    expect(r.remaining).toBe(0);
  });

  it('resets after window expires', async () => {
    await checkRateLimit(db.client, 'k', { limit: 1, windowMs: 1 });
    await new Promise((r) => setTimeout(r, 5));
    const r = await checkRateLimit(db.client, 'k', { limit: 1, windowMs: 1 });
    expect(r.allowed).toBe(true);
  });

  it('isolates limiters by key', async () => {
    await checkRateLimit(db.client, 'a', { limit: 1, windowMs: 60_000 });
    const aBlocked = await checkRateLimit(db.client, 'a', { limit: 1, windowMs: 60_000 });
    expect(aBlocked.allowed).toBe(false);
    const bAllowed = await checkRateLimit(db.client, 'b', { limit: 1, windowMs: 60_000 });
    expect(bAllowed.allowed).toBe(true);
  });
});
