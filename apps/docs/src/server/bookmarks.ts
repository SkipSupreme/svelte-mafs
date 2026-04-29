import { and, desc, eq } from 'drizzle-orm';
import { bookmark } from './schema';
import type { DB } from './db';

export async function createBookmark(
  db: DB,
  userId: string,
  lessonSlug: string,
  anchor: string | null,
): Promise<{ id: string; created: boolean }> {
  const id = crypto.randomUUID();
  // SQLite's "INSERT OR IGNORE" via Drizzle's onConflictDoNothing.
  const inserted = await db
    .insert(bookmark)
    .values({
      id,
      userId,
      lessonSlug,
      anchor,
      createdAt: new Date(),
    })
    .onConflictDoNothing({ target: [bookmark.userId, bookmark.lessonSlug, bookmark.anchor] })
    .returning({ id: bookmark.id });
  if (inserted.length > 0) return { id: inserted[0].id, created: true };
  // Already existed — fetch the original
  const existing = await db
    .select()
    .from(bookmark)
    .where(
      and(
        eq(bookmark.userId, userId),
        eq(bookmark.lessonSlug, lessonSlug),
        anchor === null ? eq(bookmark.anchor as never, null as never) : eq(bookmark.anchor, anchor),
      ),
    )
    .get();
  return { id: existing!.id, created: false };
}

export async function deleteBookmark(
  db: DB,
  userId: string,
  bookmarkId: string,
): Promise<{ deleted: boolean }> {
  const target = await db
    .select()
    .from(bookmark)
    .where(eq(bookmark.id, bookmarkId))
    .get();
  if (!target) return { deleted: false };
  if (target.userId !== userId) {
    // Foreign ownership: pretend it doesn't exist (don't leak existence).
    return { deleted: false };
  }
  await db.delete(bookmark).where(eq(bookmark.id, bookmarkId));
  return { deleted: true };
}

export async function listBookmarks(db: DB, userId: string) {
  return db
    .select()
    .from(bookmark)
    .where(eq(bookmark.userId, userId))
    .orderBy(desc(bookmark.createdAt));
}
