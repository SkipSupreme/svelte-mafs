import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { eq } from 'drizzle-orm';
import { makeTestDb, type TestDb } from '../../tests/support/d1';
import { user as userTbl } from './schema';

// Test the role-check logic directly (bypass Better Auth session creation)
async function isAdmin(db: TestDb['client'], userId: string): Promise<boolean> {
  const u = await db.select().from(userTbl).where(eq(userTbl.id, userId)).get();
  return u?.role === 'admin';
}

let db: TestDb;
beforeEach(() => { db = makeTestDb(); });
afterEach(() => db.close());

describe('admin role check', () => {
  it('returns false for users without admin role', async () => {
    const now = new Date();
    await db.client.insert(userTbl).values({
      id: 'u1',
      email: 'a@b.co',
      emailVerified: true,
      role: 'user',
      createdAt: now,
      updatedAt: now,
    });
    expect(await isAdmin(db.client, 'u1')).toBe(false);
  });

  it('returns true when role is admin', async () => {
    const now = new Date();
    await db.client.insert(userTbl).values({
      id: 'u2',
      email: 'admin@b.co',
      emailVerified: true,
      role: 'admin',
      createdAt: now,
      updatedAt: now,
    });
    expect(await isAdmin(db.client, 'u2')).toBe(true);
  });

  it('returns false for unknown user', async () => {
    expect(await isAdmin(db.client, 'nope')).toBe(false);
  });
});
