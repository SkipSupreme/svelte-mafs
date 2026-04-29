import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { and, eq } from 'drizzle-orm';
import { makeTestDb, type TestDb } from '../../tests/support/d1';
import { user as userTbl, lessonView } from './schema';
import {
  recordView,
  recordCompletion,
  mergeAnonProgress,
  getMostRecent,
  getCompletedLessons,
} from './progress';

let db: TestDb;
const USER = 'u-test';

async function seedUser() {
  const now = new Date();
  await db.client.insert(userTbl).values({
    id: USER,
    email: 'a@b.co',
    emailVerified: true,
    role: 'user',
    createdAt: now,
    updatedAt: now,
  });
}

beforeEach(async () => {
  db = makeTestDb();
  await seedUser();
});
afterEach(() => db.close());

describe('recordView', () => {
  it('inserts a new lesson_view on first call', async () => {
    await recordView(db.client, USER, {
      courseSlug: 'ml-math',
      moduleSlug: 'm5',
      lessonSlug: 'derivative',
    });
    const rows = await db.client.select().from(lessonView).all();
    expect(rows).toHaveLength(1);
    expect(rows[0].viewCount).toBe(1);
  });

  it('increments view_count and updates last_seen_at on subsequent calls', async () => {
    await recordView(db.client, USER, {
      courseSlug: 'ml-math',
      moduleSlug: 'm5',
      lessonSlug: 'derivative',
    });
    const first = await db.client
      .select()
      .from(lessonView)
      .where(and(eq(lessonView.userId, USER), eq(lessonView.lessonSlug, 'derivative')))
      .get();

    await new Promise((r) => setTimeout(r, 10));

    await recordView(db.client, USER, {
      courseSlug: 'ml-math',
      moduleSlug: 'm5',
      lessonSlug: 'derivative',
    });
    const second = await db.client
      .select()
      .from(lessonView)
      .where(and(eq(lessonView.userId, USER), eq(lessonView.lessonSlug, 'derivative')))
      .get();

    expect(second!.viewCount).toBe(2);
    expect(second!.lastSeenAt.getTime()).toBeGreaterThanOrEqual(first!.lastSeenAt.getTime());
    expect(second!.firstSeenAt.getTime()).toBe(first!.firstSeenAt.getTime());
  });

  it('isolates by user', async () => {
    const now = new Date();
    await db.client.insert(userTbl).values({
      id: 'other',
      email: 'c@d.co',
      emailVerified: true,
      role: 'user',
      createdAt: now,
      updatedAt: now,
    });
    await recordView(db.client, USER, {
      courseSlug: 'ml-math',
      moduleSlug: 'm5',
      lessonSlug: 'derivative',
    });
    await recordView(db.client, 'other', {
      courseSlug: 'ml-math',
      moduleSlug: 'm5',
      lessonSlug: 'derivative',
    });
    const rows = await db.client.select().from(lessonView).all();
    expect(rows).toHaveLength(2);
  });
});

describe('recordCompletion', () => {
  it('sets completedAt on first call', async () => {
    await recordView(db.client, USER, {
      courseSlug: 'ml-math',
      moduleSlug: 'm5',
      lessonSlug: 'derivative',
    });
    await recordCompletion(db.client, USER, 'derivative');
    const row = await db.client
      .select()
      .from(lessonView)
      .where(and(eq(lessonView.userId, USER), eq(lessonView.lessonSlug, 'derivative')))
      .get();
    expect(row!.completedAt).toBeInstanceOf(Date);
  });

  it('does not overwrite an existing completedAt', async () => {
    await recordView(db.client, USER, {
      courseSlug: 'ml-math',
      moduleSlug: 'm5',
      lessonSlug: 'derivative',
    });
    await recordCompletion(db.client, USER, 'derivative');
    const first = await db.client
      .select()
      .from(lessonView)
      .where(and(eq(lessonView.userId, USER), eq(lessonView.lessonSlug, 'derivative')))
      .get();

    await new Promise((r) => setTimeout(r, 10));
    await recordCompletion(db.client, USER, 'derivative');
    const second = await db.client
      .select()
      .from(lessonView)
      .where(and(eq(lessonView.userId, USER), eq(lessonView.lessonSlug, 'derivative')))
      .get();
    expect(second!.completedAt!.getTime()).toBe(first!.completedAt!.getTime());
  });

  it('creates a row when completing a never-viewed lesson', async () => {
    await recordCompletion(db.client, USER, 'lesson-x');
    const row = await db.client
      .select()
      .from(lessonView)
      .where(and(eq(lessonView.userId, USER), eq(lessonView.lessonSlug, 'lesson-x')))
      .get();
    expect(row!.completedAt).toBeInstanceOf(Date);
    expect(row!.viewCount).toBe(0);
  });
});

describe('mergeAnonProgress', () => {
  it('inserts new rows and skips invalid entries', async () => {
    const result = await mergeAnonProgress(db.client, USER, [
      {
        course_slug: 'ml-math',
        module_slug: 'm5',
        lesson_slug: 'derivative',
        first_seen_at: '2026-01-01T00:00:00Z',
        last_seen_at: '2026-01-01T00:01:00Z',
        view_count: 2,
      },
      // missing lesson_slug — skipped
      {
        course_slug: 'ml-math',
        module_slug: 'm5',
        lesson_slug: '',
        first_seen_at: '2026-01-01T00:00:00Z',
        last_seen_at: '2026-01-01T00:01:00Z',
      },
    ]);
    expect(result.merged).toBe(1);
  });

  it('never overwrites an existing completedAt', async () => {
    await recordView(db.client, USER, {
      courseSlug: 'ml-math',
      moduleSlug: 'm5',
      lessonSlug: 'derivative',
    });
    await recordCompletion(db.client, USER, 'derivative');

    await mergeAnonProgress(db.client, USER, [
      {
        course_slug: 'ml-math',
        module_slug: 'm5',
        lesson_slug: 'derivative',
        first_seen_at: '2026-01-01T00:00:00Z',
        last_seen_at: '2026-01-01T00:01:00Z',
        completed_at: null,
      },
    ]);

    const row = await db.client
      .select()
      .from(lessonView)
      .where(and(eq(lessonView.userId, USER), eq(lessonView.lessonSlug, 'derivative')))
      .get();
    expect(row!.completedAt).toBeInstanceOf(Date);
  });

  it('takes earliest firstSeenAt and latest lastSeenAt and sums viewCount', async () => {
    await recordView(db.client, USER, {
      courseSlug: 'ml-math',
      moduleSlug: 'm5',
      lessonSlug: 'derivative',
    });
    const original = await db.client
      .select()
      .from(lessonView)
      .where(and(eq(lessonView.userId, USER), eq(lessonView.lessonSlug, 'derivative')))
      .get();

    await mergeAnonProgress(db.client, USER, [
      {
        course_slug: 'ml-math',
        module_slug: 'm5',
        lesson_slug: 'derivative',
        first_seen_at: '2020-01-01T00:00:00Z',
        last_seen_at: '2099-01-01T00:00:00Z',
        view_count: 5,
      },
    ]);

    const merged = await db.client
      .select()
      .from(lessonView)
      .where(and(eq(lessonView.userId, USER), eq(lessonView.lessonSlug, 'derivative')))
      .get();
    expect(merged!.firstSeenAt.getUTCFullYear()).toBe(2020);
    expect(merged!.lastSeenAt.getUTCFullYear()).toBe(2099);
    expect(merged!.viewCount).toBe(original!.viewCount + 5);
  });
});

describe('getMostRecent + getCompletedLessons', () => {
  it('returns null when user has no progress in a course', async () => {
    const r = await getMostRecent(db.client, USER, 'ml-math');
    expect(r).toBeNull();
  });

  it('returns the most-recently-viewed lesson for a course', async () => {
    await recordView(db.client, USER, {
      courseSlug: 'ml-math',
      moduleSlug: 'm5',
      lessonSlug: 'derivative',
    });
    await new Promise((r) => setTimeout(r, 10));
    await recordView(db.client, USER, {
      courseSlug: 'ml-math',
      moduleSlug: 'm5',
      lessonSlug: 'chain-rule',
    });
    const r = await getMostRecent(db.client, USER, 'ml-math');
    expect(r?.lessonSlug).toBe('chain-rule');
  });

  it('lists completed slugs', async () => {
    await recordView(db.client, USER, {
      courseSlug: 'ml-math',
      moduleSlug: 'm5',
      lessonSlug: 'derivative',
    });
    await recordView(db.client, USER, {
      courseSlug: 'ml-math',
      moduleSlug: 'm5',
      lessonSlug: 'chain-rule',
    });
    await recordCompletion(db.client, USER, 'derivative');
    const set = await getCompletedLessons(db.client, USER);
    expect(set.has('derivative')).toBe(true);
    expect(set.has('chain-rule')).toBe(false);
  });
});
