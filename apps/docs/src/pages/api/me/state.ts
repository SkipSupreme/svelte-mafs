import type { APIRoute } from 'astro';
import { eq, and } from 'drizzle-orm';
import { requireSession, jsonOk } from '../../../server/middleware';
import { lessonView } from '../../../server/schema';
import { listBookmarks } from '../../../server/bookmarks';
import { listNoteIndex } from '../../../server/notes';
import { getLatestAnswers } from '../../../server/exercises';
import { getEnv } from '../../../server/env';

export const prerender = false;

export const GET: APIRoute = async ({ request, locals, url }) => {
  const env = getEnv();
  const ctx = await requireSession(request, env);
  if ('error' in ctx) return ctx.error;

  const courseSlug = url.searchParams.get('course');
  const userId = ctx.session.user.id;

  // Progress: filter by course if given
  const progressRows = courseSlug
    ? await ctx.db
        .select()
        .from(lessonView)
        .where(and(eq(lessonView.userId, userId), eq(lessonView.courseSlug, courseSlug)))
    : await ctx.db.select().from(lessonView).where(eq(lessonView.userId, userId));

  const progress = progressRows.reduce<Record<string, {
    courseSlug: string;
    moduleSlug: string;
    viewCount: number;
    completedAt: string | null;
    lastSeenAt: string;
  }>>((acc, r) => {
    acc[r.lessonSlug] = {
      courseSlug: r.courseSlug,
      moduleSlug: r.moduleSlug,
      viewCount: r.viewCount,
      completedAt: r.completedAt?.toISOString() ?? null,
      lastSeenAt: r.lastSeenAt.toISOString(),
    };
    return acc;
  }, {});

  const bookmarks = await listBookmarks(ctx.db, userId);
  const noteIndex = await listNoteIndex(ctx.db, userId);

  // Optional: per-lesson exercise state if a specific lesson is requested
  const lessonForExercises = url.searchParams.get('lesson');
  const exercises = lessonForExercises
    ? await getLatestAnswers(ctx.db, userId, lessonForExercises)
    : [];

  return jsonOk({
    user: {
      id: userId,
      email: ctx.session.user.email,
      name: ctx.session.user.name ?? null,
      image: ctx.session.user.image ?? null,
      role: ctx.session.user.role ?? 'user',
    },
    progress,
    bookmarks: bookmarks.map((b) => ({
      id: b.id,
      lessonSlug: b.lessonSlug,
      anchor: b.anchor,
      createdAt: b.createdAt.toISOString(),
    })),
    notes_index: noteIndex.map((n) => ({
      lessonSlug: n.lessonSlug,
      updatedAt: n.updatedAt.toISOString(),
    })),
    exercises,
  });
};
