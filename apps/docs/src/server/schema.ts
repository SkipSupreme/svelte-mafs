import { sqliteTable, text, integer, primaryKey, index, unique } from 'drizzle-orm/sqlite-core';

export const user = sqliteTable('user', {
  id: text('id').primaryKey(),
  email: text('email').notNull().unique(),
  emailVerified: integer('email_verified', { mode: 'boolean' }).notNull().default(false),
  name: text('name'),
  image: text('image'),
  role: text('role', { enum: ['user', 'admin'] }).notNull().default('user'),
  createdAt: integer('created_at', { mode: 'timestamp_ms' }).notNull(),
  updatedAt: integer('updated_at', { mode: 'timestamp_ms' }).notNull(),
});

export const session = sqliteTable('session', {
  id: text('id').primaryKey(),
  userId: text('user_id').notNull().references(() => user.id, { onDelete: 'cascade' }),
  token: text('token').notNull().unique(),
  expiresAt: integer('expires_at', { mode: 'timestamp_ms' }).notNull(),
  ipAddress: text('ip_address'),
  userAgent: text('user_agent'),
  createdAt: integer('created_at', { mode: 'timestamp_ms' }).notNull(),
  updatedAt: integer('updated_at', { mode: 'timestamp_ms' }).notNull(),
});

export const account = sqliteTable('account', {
  id: text('id').primaryKey(),
  userId: text('user_id').notNull().references(() => user.id, { onDelete: 'cascade' }),
  providerId: text('provider_id').notNull(),
  accountId: text('account_id').notNull(),
  accessToken: text('access_token'),
  refreshToken: text('refresh_token'),
  idToken: text('id_token'),
  accessTokenExpiresAt: integer('access_token_expires_at', { mode: 'timestamp_ms' }),
  refreshTokenExpiresAt: integer('refresh_token_expires_at', { mode: 'timestamp_ms' }),
  scope: text('scope'),
  password: text('password'),
  createdAt: integer('created_at', { mode: 'timestamp_ms' }).notNull(),
  updatedAt: integer('updated_at', { mode: 'timestamp_ms' }).notNull(),
}, (t) => ({
  uniqProvider: unique().on(t.providerId, t.accountId),
}));

export const verification = sqliteTable('verification', {
  id: text('id').primaryKey(),
  identifier: text('identifier').notNull(),
  value: text('value').notNull(),
  expiresAt: integer('expires_at', { mode: 'timestamp_ms' }).notNull(),
  createdAt: integer('created_at', { mode: 'timestamp_ms' }).notNull(),
  updatedAt: integer('updated_at', { mode: 'timestamp_ms' }).notNull(),
});

export const userProfile = sqliteTable('user_profile', {
  userId: text('user_id').primaryKey().references(() => user.id, { onDelete: 'cascade' }),
  displayName: text('display_name'),
  marketingOptIn: integer('marketing_opt_in', { mode: 'boolean' }).notNull().default(false),
  onboardedAt: integer('onboarded_at', { mode: 'timestamp_ms' }),
});

export const lessonView = sqliteTable('lesson_view', {
  userId: text('user_id').notNull().references(() => user.id, { onDelete: 'cascade' }),
  courseSlug: text('course_slug').notNull(),
  moduleSlug: text('module_slug').notNull(),
  lessonSlug: text('lesson_slug').notNull(),
  firstSeenAt: integer('first_seen_at', { mode: 'timestamp_ms' }).notNull(),
  lastSeenAt: integer('last_seen_at', { mode: 'timestamp_ms' }).notNull(),
  viewCount: integer('view_count').notNull().default(1),
  completedAt: integer('completed_at', { mode: 'timestamp_ms' }),
}, (t) => ({
  pk: primaryKey({ columns: [t.userId, t.lessonSlug] }),
  byCourseRecency: index('lesson_view_by_course_recency').on(t.userId, t.courseSlug, t.lastSeenAt),
}));

export const exerciseAnswer = sqliteTable('exercise_answer', {
  id: text('id').primaryKey(),
  userId: text('user_id').notNull().references(() => user.id, { onDelete: 'cascade' }),
  lessonSlug: text('lesson_slug').notNull(),
  exerciseId: text('exercise_id').notNull(),
  answerJson: text('answer_json').notNull(),
  isCorrect: integer('is_correct', { mode: 'boolean' }),
  attemptNo: integer('attempt_no').notNull().default(1),
  createdAt: integer('created_at', { mode: 'timestamp_ms' }).notNull(),
}, (t) => ({
  byUserLesson: index('exercise_answer_by_user_lesson').on(t.userId, t.lessonSlug),
}));

export const bookmark = sqliteTable('bookmark', {
  id: text('id').primaryKey(),
  userId: text('user_id').notNull().references(() => user.id, { onDelete: 'cascade' }),
  lessonSlug: text('lesson_slug').notNull(),
  anchor: text('anchor'),
  createdAt: integer('created_at', { mode: 'timestamp_ms' }).notNull(),
}, (t) => ({
  uniq: unique().on(t.userId, t.lessonSlug, t.anchor),
}));

export const note = sqliteTable('note', {
  userId: text('user_id').notNull().references(() => user.id, { onDelete: 'cascade' }),
  lessonSlug: text('lesson_slug').notNull(),
  body: text('body').notNull().default(''),
  updatedAt: integer('updated_at', { mode: 'timestamp_ms' }).notNull(),
}, (t) => ({
  pk: primaryKey({ columns: [t.userId, t.lessonSlug] }),
}));

export const emailDrop = sqliteTable('email_drop', {
  id: text('id').primaryKey(),
  subject: text('subject').notNull(),
  bodyMd: text('body_md').notNull(),
  courseSlug: text('course_slug'),
  moduleSlug: text('module_slug'),
  lessonSlug: text('lesson_slug'),
  targetCount: integer('target_count').notNull(),
  sentCount: integer('sent_count').notNull(),
  sentAt: integer('sent_at', { mode: 'timestamp_ms' }).notNull(),
  sentByUserId: text('sent_by_user_id').references(() => user.id),
});

export const rateLimit = sqliteTable('rate_limit', {
  key: text('key').primaryKey(),
  count: integer('count').notNull(),
  resetAt: integer('reset_at', { mode: 'timestamp_ms' }).notNull(),
});
