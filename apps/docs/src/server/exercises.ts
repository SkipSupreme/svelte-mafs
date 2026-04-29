import { and, desc, eq, max } from 'drizzle-orm';
import { exerciseAnswer } from './schema';
import type { DB } from './db';

export interface ExerciseSubmission {
  lessonSlug: string;
  exerciseId: string;
  answerJson: unknown;
  isCorrect: boolean | null;
}

export async function recordExerciseAnswer(
  db: DB,
  userId: string,
  s: ExerciseSubmission,
): Promise<{ id: string; attemptNo: number }> {
  // Find the highest attemptNo for this (user, lesson, exercise)
  const prev = await db
    .select({ n: max(exerciseAnswer.attemptNo) })
    .from(exerciseAnswer)
    .where(
      and(
        eq(exerciseAnswer.userId, userId),
        eq(exerciseAnswer.lessonSlug, s.lessonSlug),
        eq(exerciseAnswer.exerciseId, s.exerciseId),
      ),
    )
    .get();
  const attemptNo = (prev?.n ?? 0) + 1;
  const id = crypto.randomUUID();
  await db.insert(exerciseAnswer).values({
    id,
    userId,
    lessonSlug: s.lessonSlug,
    exerciseId: s.exerciseId,
    answerJson: JSON.stringify(s.answerJson),
    isCorrect: s.isCorrect,
    attemptNo,
    createdAt: new Date(),
  });
  return { id, attemptNo };
}

export async function getLatestAnswers(
  db: DB,
  userId: string,
  lessonSlug: string,
): Promise<Array<{ exerciseId: string; answerJson: unknown; isCorrect: boolean | null; attemptNo: number; createdAt: Date }>> {
  const rows = await db
    .select()
    .from(exerciseAnswer)
    .where(and(eq(exerciseAnswer.userId, userId), eq(exerciseAnswer.lessonSlug, lessonSlug)))
    .orderBy(desc(exerciseAnswer.attemptNo), desc(exerciseAnswer.createdAt));
  // Keep only the latest per exerciseId
  const seen = new Set<string>();
  const out: Array<{ exerciseId: string; answerJson: unknown; isCorrect: boolean | null; attemptNo: number; createdAt: Date }> = [];
  for (const r of rows) {
    if (seen.has(r.exerciseId)) continue;
    seen.add(r.exerciseId);
    out.push({
      exerciseId: r.exerciseId,
      answerJson: safeJsonParse(r.answerJson),
      isCorrect: r.isCorrect,
      attemptNo: r.attemptNo,
      createdAt: r.createdAt,
    });
  }
  return out;
}

function safeJsonParse(s: string): unknown {
  try {
    return JSON.parse(s);
  } catch {
    return null;
  }
}
