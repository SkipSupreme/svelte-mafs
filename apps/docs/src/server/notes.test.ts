import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { makeTestDb, type TestDb } from '../../tests/support/d1';
import { user as userTbl } from './schema';
import { upsertNote, getNote, listNoteIndex } from './notes';

let db: TestDb;
const USER = 'u-notes';

beforeEach(async () => {
  db = makeTestDb();
  const now = new Date();
  await db.client.insert(userTbl).values({
    id: USER,
    email: 'a@b.co',
    emailVerified: true,
    role: 'user',
    createdAt: now,
    updatedAt: now,
  });
});
afterEach(() => db.close());

describe('notes', () => {
  it('upsert + get round-trip', async () => {
    await upsertNote(db.client, USER, 'derivative', 'hello world');
    const r = await getNote(db.client, USER, 'derivative');
    expect(r?.body).toBe('hello world');
  });

  it('overwrites body and updates timestamp', async () => {
    await upsertNote(db.client, USER, 'derivative', 'first');
    const a = await getNote(db.client, USER, 'derivative');
    await new Promise((r) => setTimeout(r, 5));
    await upsertNote(db.client, USER, 'derivative', 'second');
    const b = await getNote(db.client, USER, 'derivative');
    expect(b?.body).toBe('second');
    expect(b!.updatedAt.getTime()).toBeGreaterThan(a!.updatedAt.getTime());
  });

  it('returns null for missing note', async () => {
    const r = await getNote(db.client, USER, 'nope');
    expect(r).toBeNull();
  });

  it('lists note index for user', async () => {
    await upsertNote(db.client, USER, 'a', 'x');
    await upsertNote(db.client, USER, 'b', 'y');
    const list = await listNoteIndex(db.client, USER);
    expect(list).toHaveLength(2);
    expect(list.map((r) => r.lessonSlug).sort()).toEqual(['a', 'b']);
  });
});
