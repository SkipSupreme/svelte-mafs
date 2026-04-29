import { and, eq } from 'drizzle-orm';
import { note } from './schema';
import type { DB } from './db';

export const NOTE_MAX_BYTES = 50_000;

export async function upsertNote(
  db: DB,
  userId: string,
  lessonSlug: string,
  body: string,
): Promise<{ updatedAt: Date }> {
  const updatedAt = new Date();
  await db
    .insert(note)
    .values({ userId, lessonSlug, body, updatedAt })
    .onConflictDoUpdate({
      target: [note.userId, note.lessonSlug],
      set: { body, updatedAt },
    });
  return { updatedAt };
}

export async function getNote(
  db: DB,
  userId: string,
  lessonSlug: string,
): Promise<{ body: string; updatedAt: Date } | null> {
  const row = await db
    .select()
    .from(note)
    .where(and(eq(note.userId, userId), eq(note.lessonSlug, lessonSlug)))
    .get();
  if (!row) return null;
  return { body: row.body, updatedAt: row.updatedAt };
}

export async function listNoteIndex(
  db: DB,
  userId: string,
): Promise<Array<{ lessonSlug: string; updatedAt: Date }>> {
  const rows = await db
    .select({ lessonSlug: note.lessonSlug, updatedAt: note.updatedAt })
    .from(note)
    .where(eq(note.userId, userId));
  return rows;
}
