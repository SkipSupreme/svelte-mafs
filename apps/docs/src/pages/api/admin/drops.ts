import type { APIRoute } from 'astro';
import { eq } from 'drizzle-orm';
import { z } from 'zod';
import { requireCsrf, jsonError, jsonOk } from '../../../server/middleware';
import { requireAdmin } from '../../../server/admin';
import { user as userTbl, userProfile, emailDrop } from '../../../server/schema';
import { sendDropEmail, signUnsubscribeToken } from '../../../server/email';
import { getEnv } from '../../../server/env';

export const prerender = false;

const Body = z.object({
  subject: z.string().min(1).max(200),
  body_html: z.string().min(1).max(100_000),
  lesson_slug: z.string().max(200).optional(),
  course_slug: z.string().max(100).optional(),
  module_slug: z.string().max(100).optional(),
  test_only: z.boolean().default(false),
});

interface AudienceRow {
  userId: string;
  email: string;
}

export const POST: APIRoute = async ({ request, locals }) => {
  const env = getEnv();
  const csrf = requireCsrf(request);
  if (csrf) return csrf;

  const guard = await requireAdmin(request, {
    DB: env.DB,
    BETTER_AUTH_SECRET: env.BETTER_AUTH_SECRET,
    GOOGLE_CLIENT_ID: env.GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET: env.GOOGLE_CLIENT_SECRET,
    GITHUB_CLIENT_ID: env.GITHUB_CLIENT_ID,
    GITHUB_CLIENT_SECRET: env.GITHUB_CLIENT_SECRET,
    RESEND_API_KEY: env.RESEND_API_KEY,
    PUBLIC_SITE_URL: env.PUBLIC_SITE_URL,
  });
  if (!guard.ok) return guard.response;

  const { db, userId: adminId, email: adminEmail } = guard;

  if (!env.RESEND_API_KEY) {
    return jsonError(503, 'no_resend', 'RESEND_API_KEY not configured');
  }
  if (!env.UNSUBSCRIBE_HMAC_SECRET) {
    return jsonError(503, 'no_hmac', 'UNSUBSCRIBE_HMAC_SECRET not configured');
  }

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return jsonError(400, 'bad_json', 'Could not parse JSON body');
  }
  const parsed = Body.safeParse(body);
  if (!parsed.success) return jsonError(400, 'bad_request', 'Invalid payload');

  let audience: AudienceRow[];
  if (parsed.data.test_only) {
    audience = [{ userId: adminId, email: adminEmail }];
  } else {
    audience = await db
      .select({ userId: userTbl.id, email: userTbl.email })
      .from(userTbl)
      .innerJoin(userProfile, eq(userProfile.userId, userTbl.id))
      .where(eq(userProfile.marketingOptIn, true));
  }

  const dropId = crypto.randomUUID();
  const now = new Date();
  await db.insert(emailDrop).values({
    id: dropId,
    subject: parsed.data.subject,
    bodyMd: parsed.data.body_html,
    courseSlug: parsed.data.course_slug ?? null,
    moduleSlug: parsed.data.module_slug ?? null,
    lessonSlug: parsed.data.lesson_slug ?? null,
    targetCount: audience.length,
    sentCount: 0,
    sentAt: now,
    sentByUserId: adminId,
  });

  let sentCount = 0;
  const failures: string[] = [];
  for (const recipient of audience) {
    try {
      const token = await signUnsubscribeToken(recipient.userId, env.UNSUBSCRIBE_HMAC_SECRET);
      const unsubUrl = `${env.PUBLIC_SITE_URL}/api/unsubscribe?token=${encodeURIComponent(token)}`;
      await sendDropEmail(
        env.RESEND_API_KEY,
        recipient.email,
        parsed.data.subject,
        parsed.data.body_html,
        unsubUrl,
      );
      sentCount++;
    } catch (e) {
      failures.push(`${recipient.email}: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  await db.update(emailDrop).set({ sentCount }).where(eq(emailDrop.id, dropId));

  return jsonOk({
    id: dropId,
    target_count: audience.length,
    sent_count: sentCount,
    failures: failures.length ? failures.slice(0, 10) : undefined,
  });
};
