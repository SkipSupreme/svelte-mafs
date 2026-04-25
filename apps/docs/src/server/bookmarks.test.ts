import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { makeTestDb, type TestDb } from '../../tests/support/d1';
import { user as userTbl } from './schema';
import { createBookmark, deleteBookmark, listBookmarks } from './bookmarks';

let db: TestDb;
const USER = 'u-bookmark';
const OTHER = 'u-other';

beforeEach(async () => {
  db = makeTestDb();
  const now = new Date();
  await db.client.insert(userTbl).values([
    { id: USER, email: 'a@b.co', emailVerified: true, role: 'user', createdAt: now, updatedAt: now },
    { id: OTHER, email: 'c@d.co', emailVerified: true, role: 'user', createdAt: now, updatedAt: now },
  ]);
});
afterEach(() => db.close());

describe('createBookmark', () => {
  it('creates a bookmark', async () => {
    const r = await createBookmark(db.client, USER, 'derivative', null);
    expect(r.created).toBe(true);
    expect(r.id).toBeTruthy();
  });

  it('is idempotent for same (lesson, anchor)', async () => {
    const a = await createBookmark(db.client, USER, 'derivative', '#section-1');
    const b = await createBookmark(db.client, USER, 'derivative', '#section-1');
    expect(b.created).toBe(false);
    expect(b.id).toBe(a.id);
  });
});

describe('deleteBookmark', () => {
  it('deletes when owned by user', async () => {
    const a = await createBookmark(db.client, USER, 'derivative', null);
    const r = await deleteBookmark(db.client, USER, a.id);
    expect(r.deleted).toBe(true);
    const list = await listBookmarks(db.client, USER);
    expect(list).toHaveLength(0);
  });

  it('refuses to delete another user\'s bookmark (returns deleted:false)', async () => {
    const a = await createBookmark(db.client, USER, 'derivative', null);
    const r = await deleteBookmark(db.client, OTHER, a.id);
    expect(r.deleted).toBe(false);
    const list = await listBookmarks(db.client, USER);
    expect(list).toHaveLength(1); // still there
  });

  it('returns deleted:false for unknown id', async () => {
    const r = await deleteBookmark(db.client, USER, 'nope');
    expect(r.deleted).toBe(false);
  });
});

describe('listBookmarks', () => {
  it('returns user\'s bookmarks newest-first', async () => {
    await createBookmark(db.client, USER, 'a', null);
    await new Promise((r) => setTimeout(r, 5));
    await createBookmark(db.client, USER, 'b', null);
    const list = await listBookmarks(db.client, USER);
    expect(list).toHaveLength(2);
    expect(list[0].lessonSlug).toBe('b');
  });
});
