import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { makeTestDb, type TestDb } from '../../tests/support/d1';
import { user as userTbl } from './schema';
import { recordExerciseAnswer, getLatestAnswers } from './exercises';

let db: TestDb;
const USER = 'u-exercise';

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

describe('recordExerciseAnswer', () => {
  it('starts attempt_no at 1', async () => {
    const r = await recordExerciseAnswer(db.client, USER, {
      lessonSlug: 'derivative',
      exerciseId: 'q1',
      answerJson: { value: 42 },
      isCorrect: true,
    });
    expect(r.attemptNo).toBe(1);
  });

  it('increments attempt_no on subsequent attempts', async () => {
    await recordExerciseAnswer(db.client, USER, {
      lessonSlug: 'derivative',
      exerciseId: 'q1',
      answerJson: { v: 1 },
      isCorrect: false,
    });
    const r2 = await recordExerciseAnswer(db.client, USER, {
      lessonSlug: 'derivative',
      exerciseId: 'q1',
      answerJson: { v: 2 },
      isCorrect: true,
    });
    expect(r2.attemptNo).toBe(2);
  });

  it('isolates attempt_no per (user, lesson, exercise)', async () => {
    await recordExerciseAnswer(db.client, USER, {
      lessonSlug: 'derivative',
      exerciseId: 'q1',
      answerJson: {},
      isCorrect: false,
    });
    const otherExercise = await recordExerciseAnswer(db.client, USER, {
      lessonSlug: 'derivative',
      exerciseId: 'q2',
      answerJson: {},
      isCorrect: false,
    });
    expect(otherExercise.attemptNo).toBe(1);
  });
});

describe('getLatestAnswers', () => {
  it('returns latest attempt per exercise', async () => {
    await recordExerciseAnswer(db.client, USER, {
      lessonSlug: 'derivative',
      exerciseId: 'q1',
      answerJson: { v: 1 },
      isCorrect: false,
    });
    await recordExerciseAnswer(db.client, USER, {
      lessonSlug: 'derivative',
      exerciseId: 'q1',
      answerJson: { v: 2 },
      isCorrect: true,
    });
    await recordExerciseAnswer(db.client, USER, {
      lessonSlug: 'derivative',
      exerciseId: 'q2',
      answerJson: { v: 'x' },
      isCorrect: null,
    });
    const latest = await getLatestAnswers(db.client, USER, 'derivative');
    expect(latest).toHaveLength(2);
    const q1 = latest.find((r) => r.exerciseId === 'q1');
    expect(q1?.answerJson).toEqual({ v: 2 });
    expect(q1?.attemptNo).toBe(2);
    expect(q1?.isCorrect).toBe(true);
  });
});
