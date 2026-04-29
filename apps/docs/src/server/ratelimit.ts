import { eq } from 'drizzle-orm';
import { rateLimit } from './schema';
import type { DB } from './db';

export interface RateLimitResult {
  allowed: boolean;
  remaining: number;
  retryAfterMs: number;
}

export interface RateLimitOpts {
  limit: number;
  windowMs: number;
}

/**
 * Single-row sliding-window limiter keyed by an arbitrary string.
 * Idempotent: safe to call from concurrent requests; the worst case
 * is undercounting by one in a tight race, which is acceptable for
 * abuse mitigation (not for billing).
 */
export async function checkRateLimit(
  db: DB,
  key: string,
  opts: RateLimitOpts,
): Promise<RateLimitResult> {
  const now = new Date();
  const row = await db.select().from(rateLimit).where(eq(rateLimit.key, key)).get();

  if (!row || row.resetAt.getTime() <= now.getTime()) {
    const resetAt = new Date(now.getTime() + opts.windowMs);
    await db
      .insert(rateLimit)
      .values({ key, count: 1, resetAt })
      .onConflictDoUpdate({
        target: rateLimit.key,
        set: { count: 1, resetAt },
      });
    return { allowed: true, remaining: opts.limit - 1, retryAfterMs: 0 };
  }

  if (row.count >= opts.limit) {
    return {
      allowed: false,
      remaining: 0,
      retryAfterMs: row.resetAt.getTime() - now.getTime(),
    };
  }

  await db
    .update(rateLimit)
    .set({ count: row.count + 1 })
    .where(eq(rateLimit.key, key));
  return {
    allowed: true,
    remaining: opts.limit - 1 - row.count,
    retryAfterMs: 0,
  };
}
