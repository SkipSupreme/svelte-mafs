import { and, desc, eq, sql } from 'drizzle-orm';
import { lessonView } from './schema';
import type { DB } from './db';

export interface ViewInput {
  courseSlug: string;
  moduleSlug: string;
  lessonSlug: string;
}

/** Upsert a view record. Idempotent; safe under concurrent calls. */
export async function recordView(db: DB, userId: string, input: ViewInput): Promise<void> {
  const now = new Date();
  await db
    .insert(lessonView)
    .values({
      userId,
      courseSlug: input.courseSlug,
      moduleSlug: input.moduleSlug,
      lessonSlug: input.lessonSlug,
      firstSeenAt: now,
      lastSeenAt: now,
      viewCount: 1,
    })
    .onConflictDoUpdate({
      target: [lessonView.userId, lessonView.lessonSlug],
      set: {
        lastSeenAt: now,
        viewCount: sql`${lessonView.viewCount} + 1`,
      },
    });
}

/** Mark a lesson complete. Only sets completedAt if currently null (monotonic). */
export async function recordCompletion(
  db: DB,
  userId: string,
  lessonSlug: string,
): Promise<void> {
  const now = new Date();
  const existing = await db
    .select()
    .from(lessonView)
    .where(and(eq(lessonView.userId, userId), eq(lessonView.lessonSlug, lessonSlug)))
    .get();

  if (!existing) {
    // Mark complete on a lesson with no prior view record. Synthesize
    // course/module slugs as empty strings; the next view event from the
    // page will fill them in.
    await db.insert(lessonView).values({
      userId,
      courseSlug: '',
      moduleSlug: '',
      lessonSlug,
      firstSeenAt: now,
      lastSeenAt: now,
      viewCount: 0,
      completedAt: now,
    });
    return;
  }

  if (existing.completedAt) return;

  await db
    .update(lessonView)
    .set({ completedAt: now, lastSeenAt: now })
    .where(and(eq(lessonView.userId, userId), eq(lessonView.lessonSlug, lessonSlug)));
}

export interface AnonProgressEntry {
  course_slug: string;
  module_slug: string;
  lesson_slug: string;
  first_seen_at: string;
  last_seen_at: string;
  view_count?: number;
  completed_at?: string | null;
}

/**
 * Merge anonymous (localStorage) progress into a user's account.
 * Idempotent: never overwrites an existing completedAt; takes the
 * earlier firstSeenAt and the later lastSeenAt; sums viewCount.
 */
export async function mergeAnonProgress(
  db: DB,
  userId: string,
  entries: AnonProgressEntry[],
): Promise<{ merged: number }> {
  let merged = 0;
  for (const e of entries) {
    if (!e.lesson_slug) continue;
    const existing = await db
      .select()
      .from(lessonView)
      .where(and(eq(lessonView.userId, userId), eq(lessonView.lessonSlug, e.lesson_slug)))
      .get();

    const firstSeen = parseDate(e.first_seen_at);
    const lastSeen = parseDate(e.last_seen_at);
    if (!firstSeen || !lastSeen) continue;

    if (!existing) {
      await db.insert(lessonView).values({
        userId,
        courseSlug: e.course_slug,
        moduleSlug: e.module_slug,
        lessonSlug: e.lesson_slug,
        firstSeenAt: firstSeen,
        lastSeenAt: lastSeen,
        viewCount: e.view_count ?? 1,
        completedAt: parseDate(e.completed_at) ?? null,
      });
    } else {
      const newer = lastSeen.getTime() > existing.lastSeenAt.getTime() ? lastSeen : existing.lastSeenAt;
      const older =
        firstSeen.getTime() < existing.firstSeenAt.getTime() ? firstSeen : existing.firstSeenAt;
      const incomingCompletion = parseDate(e.completed_at);
      // Existing completedAt wins (never overwrite); else fall back to incoming.
      const completedAt = existing.completedAt ?? incomingCompletion ?? null;
      await db
        .update(lessonView)
        .set({
          firstSeenAt: older,
          lastSeenAt: newer,
          viewCount: existing.viewCount + (e.view_count ?? 1),
          completedAt,
        })
        .where(and(eq(lessonView.userId, userId), eq(lessonView.lessonSlug, e.lesson_slug)));
    }
    merged++;
  }
  return { merged };
}

function parseDate(s: string | null | undefined): Date | null {
  if (!s) return null;
  const d = new Date(s);
  return isNaN(d.getTime()) ? null : d;
}

export interface ContinueLesson {
  courseSlug: string;
  moduleSlug: string;
  lessonSlug: string;
  lastSeenAt: Date;
}

export async function getMostRecent(
  db: DB,
  userId: string,
  courseSlug: string,
): Promise<ContinueLesson | null> {
  const row = await db
    .select()
    .from(lessonView)
    .where(and(eq(lessonView.userId, userId), eq(lessonView.courseSlug, courseSlug)))
    .orderBy(desc(lessonView.lastSeenAt))
    .limit(1)
    .get();
  if (!row) return null;
  return {
    courseSlug: row.courseSlug,
    moduleSlug: row.moduleSlug,
    lessonSlug: row.lessonSlug,
    lastSeenAt: row.lastSeenAt,
  };
}

export async function getCompletedLessons(db: DB, userId: string): Promise<Set<string>> {
  const rows = await db
    .select({ slug: lessonView.lessonSlug, completedAt: lessonView.completedAt })
    .from(lessonView)
    .where(eq(lessonView.userId, userId));
  return new Set(rows.filter((r) => r.completedAt).map((r) => r.slug));
}
