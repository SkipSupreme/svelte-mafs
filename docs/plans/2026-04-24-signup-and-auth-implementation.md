# Signup & Auth v1 — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Replace the stub waitlist with a real signup flow. Build auth (magic link + Google + GitHub), progress tracking, bookmarks, notes, persisted exercises, and an admin UI for "new module live" emails — all on Cloudflare D1 + Better Auth + Resend.

**Architecture:** Astro switches from `output: 'static'` to `output: 'server'` with per-page `prerender = true`. Auth + product data live in Cloudflare D1 via Drizzle. Better Auth handles all `/api/auth/*` flows. Resend sends transactional + marketing email. Admin UI gated by `user.role = 'admin'` and 404s for non-admins. All endpoints are session-required, CSRF-protected, Zod-validated, and rate-limited.

**Tech Stack:** Astro 6 · Svelte 5 · Cloudflare Workers · Cloudflare D1 · Drizzle ORM · Better Auth · Resend · Zod · Cloudflare Turnstile · Vitest · Playwright

**Companion design:** `docs/plans/2026-04-24-signup-and-auth-design.md` — read first if any task feels under-specified.

**Worktree:** `.worktrees/feature-auth-v1` on branch `feature/auth-v1`. Merge to `main` after Phase 11 completes.

---

## Pre-flight checklist (owner-only, out-of-band)

These cannot be automated. Confirm complete before starting Phase 2 ("Auth backbone").

- [ ] **Google OAuth app** registered at https://console.cloud.google.com/apis/credentials. Authorized redirect URIs: `https://learntinker.com/api/auth/callback/google` AND `http://localhost:4321/api/auth/callback/google`. Store `client_id` + `client_secret`.
- [ ] **GitHub OAuth app** registered at https://github.com/settings/developers. Same callback URLs. Store `client_id` + `client_secret`.
- [ ] **Resend account** at https://resend.com. Domain `learntinker.com` verified (DKIM/SPF/DMARC records added to Cloudflare DNS). API key generated.
- [ ] **Cloudflare Turnstile site** created in dashboard (invisible mode). Store site key + secret key.
- [ ] **Generated secrets:** `BETTER_AUTH_SECRET=$(openssl rand -hex 32)` and `UNSUBSCRIBE_HMAC_SECRET=$(openssl rand -hex 32)`.

Once all check, save secrets to `apps/docs/.dev.vars` (gitignored, format `KEY=value` one per line) AND set in Cloudflare via `wrangler secret put <NAME>` (one at a time, paste when prompted).

---

# Phase 1 — Foundation

Goal: D1 binding, Drizzle schema, server-output Astro. Zero behavior change for users.

## Task 1.1: Add D1 binding to wrangler.jsonc

**Files:**
- Modify: `apps/docs/wrangler.jsonc`

**Step 1: Add the binding.** Open `apps/docs/wrangler.jsonc` and add a `d1_databases` block above `routes`:

```jsonc
"d1_databases": [
  {
    "binding": "DB",
    "database_name": "tinker",
    "database_id": "REPLACE_AFTER_CREATE",
    "migrations_dir": "migrations"
  }
],
```

**Step 2: Create the actual D1 instance.**

Run: `cd apps/docs && pnpm dlx wrangler@latest d1 create tinker`
Expected: Output includes a UUID. Copy it into `database_id` in `wrangler.jsonc`, replacing `REPLACE_AFTER_CREATE`.

**Step 3: Commit.**

```bash
git add apps/docs/wrangler.jsonc
git commit -m "feat(infra): bind Cloudflare D1 'tinker' to docs Worker"
```

---

## Task 1.2: Switch Astro to server output

**Files:**
- Modify: `apps/docs/astro.config.mjs`
- Modify: every `.astro` page under `apps/docs/src/pages/**`

**Step 1: Flip the output mode.** Edit `apps/docs/astro.config.mjs`:

```js
export default defineConfig({
  // ...existing config...
  output: 'server',  // was 'static'
  // ...
});
```

**Step 2: Add `prerender = true` to every existing page.** For each `.astro` file under `apps/docs/src/pages/`, add a frontmatter block at the top:

```astro
---
export const prerender = true;
// existing frontmatter below
---
```

Files to touch: `index.astro`, `404.astro`, `courses/[course]/index.astro`. Any MDX page under `pages/` already prerenders by default — no change.

**Step 3: Verify build still works.**

Run: `cd apps/docs && pnpm build`
Expected: Build succeeds. Output mentions `(SSR)` for the worker handler but every page is prerendered.

**Step 4: Commit.**

```bash
git add apps/docs/astro.config.mjs apps/docs/src/pages/
git commit -m "feat(infra): astro output:'server', prerender existing pages"
```

---

## Task 1.3: Install dependencies

**Files:** `apps/docs/package.json`, `pnpm-lock.yaml`

**Step 1: Install runtime deps.**

```bash
cd apps/docs && pnpm add better-auth drizzle-orm resend zod
```

**Step 2: Install dev deps.**

```bash
pnpm add -D drizzle-kit @cloudflare/workers-types
```

**Step 3: Verify versions resolved.**

Run: `cat package.json | grep -E "(better-auth|drizzle|resend|zod)"`
Expected: Each package present with a version.

**Step 4: Commit.**

```bash
git add package.json pnpm-lock.yaml ../../pnpm-lock.yaml
git commit -m "feat(infra): add better-auth, drizzle, resend, zod"
```

---

## Task 1.4: Drizzle schema (auth + product tables)

**Files:**
- Create: `apps/docs/src/server/schema.ts`
- Create: `apps/docs/drizzle.config.ts`

**Step 1: Write the schema.** Full schema from design doc Section "Data model" goes into `apps/docs/src/server/schema.ts`. Use Drizzle's SQLite syntax:

```ts
import { sqliteTable, text, integer, primaryKey, index, unique } from 'drizzle-orm/sqlite-core';

export const user = sqliteTable('user', {
  id: text('id').primaryKey(),
  email: text('email').notNull().unique(),
  emailVerified: integer('email_verified', { mode: 'boolean' }).notNull().default(false),
  name: text('name'),
  image: text('image'),
  role: text('role', { enum: ['user', 'admin'] }).notNull().default('user'),
  createdAt: integer('created_at', { mode: 'timestamp' }).notNull(),
  updatedAt: integer('updated_at', { mode: 'timestamp' }).notNull(),
});

export const session = sqliteTable('session', {
  id: text('id').primaryKey(),
  userId: text('user_id').notNull().references(() => user.id, { onDelete: 'cascade' }),
  token: text('token').notNull().unique(),
  expiresAt: integer('expires_at', { mode: 'timestamp' }).notNull(),
  ip: text('ip'),
  userAgent: text('user_agent'),
  createdAt: integer('created_at', { mode: 'timestamp' }).notNull(),
});

export const account = sqliteTable('account', {
  id: text('id').primaryKey(),
  userId: text('user_id').notNull().references(() => user.id, { onDelete: 'cascade' }),
  providerId: text('provider_id').notNull(),
  accountId: text('account_id').notNull(),
}, (t) => ({
  uniqProvider: unique().on(t.providerId, t.accountId),
}));

export const verification = sqliteTable('verification', {
  id: text('id').primaryKey(),
  identifier: text('identifier').notNull(),
  value: text('value').notNull(),
  expiresAt: integer('expires_at', { mode: 'timestamp' }).notNull(),
  createdAt: integer('created_at', { mode: 'timestamp' }).notNull(),
});

export const userProfile = sqliteTable('user_profile', {
  userId: text('user_id').primaryKey().references(() => user.id, { onDelete: 'cascade' }),
  displayName: text('display_name'),
  marketingOptIn: integer('marketing_opt_in', { mode: 'boolean' }).notNull().default(false),
  onboardedAt: integer('onboarded_at', { mode: 'timestamp' }),
});

export const lessonView = sqliteTable('lesson_view', {
  userId: text('user_id').notNull().references(() => user.id, { onDelete: 'cascade' }),
  courseSlug: text('course_slug').notNull(),
  moduleSlug: text('module_slug').notNull(),
  lessonSlug: text('lesson_slug').notNull(),
  firstSeenAt: integer('first_seen_at', { mode: 'timestamp' }).notNull(),
  lastSeenAt: integer('last_seen_at', { mode: 'timestamp' }).notNull(),
  viewCount: integer('view_count').notNull().default(1),
  completedAt: integer('completed_at', { mode: 'timestamp' }),
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
  createdAt: integer('created_at', { mode: 'timestamp' }).notNull(),
}, (t) => ({
  byUserLesson: index('exercise_answer_by_user_lesson').on(t.userId, t.lessonSlug),
}));

export const bookmark = sqliteTable('bookmark', {
  id: text('id').primaryKey(),
  userId: text('user_id').notNull().references(() => user.id, { onDelete: 'cascade' }),
  lessonSlug: text('lesson_slug').notNull(),
  anchor: text('anchor'),
  createdAt: integer('created_at', { mode: 'timestamp' }).notNull(),
}, (t) => ({
  uniq: unique().on(t.userId, t.lessonSlug, t.anchor),
}));

export const note = sqliteTable('note', {
  userId: text('user_id').notNull().references(() => user.id, { onDelete: 'cascade' }),
  lessonSlug: text('lesson_slug').notNull(),
  body: text('body').notNull().default(''),
  updatedAt: integer('updated_at', { mode: 'timestamp' }).notNull(),
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
  sentAt: integer('sent_at', { mode: 'timestamp' }).notNull(),
  sentByUserId: text('sent_by_user_id').references(() => user.id),
});
```

**Step 2: Drizzle config.** `apps/docs/drizzle.config.ts`:

```ts
import { defineConfig } from 'drizzle-kit';

export default defineConfig({
  schema: './src/server/schema.ts',
  out: './migrations',
  dialect: 'sqlite',
  driver: 'd1-http',
});
```

**Step 3: Generate migration.**

Run: `cd apps/docs && pnpm drizzle-kit generate`
Expected: Creates `apps/docs/migrations/0000_*.sql`. Inspect the SQL — confirm all 10 tables present.

**Step 4: Apply migration locally.**

Run: `pnpm dlx wrangler@latest d1 migrations apply tinker --local`
Expected: "Migrations applied successfully."

**Step 5: Apply migration to remote (only after pre-flight Resend/OAuth done).**

For now skip this. Will run in Phase 11.

**Step 6: Commit.**

```bash
git add apps/docs/src/server/schema.ts apps/docs/drizzle.config.ts apps/docs/migrations/
git commit -m "feat(db): drizzle schema for auth + product tables"
```

---

## Task 1.5: DB client factory

**Files:**
- Create: `apps/docs/src/server/db.ts`
- Test: `apps/docs/src/server/db.test.ts`

**Step 1: Write the failing test.**

```ts
// db.test.ts
import { describe, it, expect } from 'vitest';
import { getDb } from './db';

describe('getDb', () => {
  it('returns a drizzle client when given a D1 binding', () => {
    const fakeBinding = { prepare: () => ({}) } as unknown as D1Database;
    const db = getDb(fakeBinding);
    expect(db).toBeDefined();
    expect(typeof db.select).toBe('function');
  });

  it('throws if binding is missing', () => {
    expect(() => getDb(undefined as unknown as D1Database)).toThrow(/D1 binding/);
  });
});
```

**Step 2: Run test, verify failure.**

Run: `cd apps/docs && pnpm vitest run src/server/db.test.ts`
Expected: FAIL — module not found.

**Step 3: Implement.**

```ts
// db.ts
import { drizzle } from 'drizzle-orm/d1';
import * as schema from './schema';

export type DB = ReturnType<typeof drizzle<typeof schema>>;

export function getDb(binding: D1Database): DB {
  if (!binding) throw new Error('D1 binding missing — check wrangler.jsonc and Astro.locals.runtime.env');
  return drizzle(binding, { schema });
}
```

**Step 4: Add vitest config and run.** Create `apps/docs/vitest.config.ts`:

```ts
import { defineConfig } from 'vitest/config';
export default defineConfig({
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
});
```

Add `"test": "vitest run"` to `apps/docs/package.json` scripts. Run: `pnpm test`
Expected: 2 passed.

**Step 5: Commit.**

```bash
git add apps/docs/src/server/db.ts apps/docs/src/server/db.test.ts apps/docs/vitest.config.ts apps/docs/package.json
git commit -m "feat(db): drizzle client factory"
```

---

# Phase 2 — Auth backbone

Goal: working Better Auth with magic link + Google + GitHub. Sessions in D1. No UI yet.

## Task 2.1: Better Auth config

**Files:**
- Create: `apps/docs/src/server/auth.ts`
- Test: `apps/docs/src/server/auth.test.ts`

**Step 1: Write the failing test.**

```ts
import { describe, it, expect } from 'vitest';
import { createAuth } from './auth';

const fakeEnv = {
  DB: { prepare: () => ({}) } as unknown as D1Database,
  BETTER_AUTH_SECRET: 'test-secret-32-bytes-of-entropy-yes',
  GOOGLE_CLIENT_ID: 'g-id',
  GOOGLE_CLIENT_SECRET: 'g-secret',
  GITHUB_CLIENT_ID: 'gh-id',
  GITHUB_CLIENT_SECRET: 'gh-secret',
  RESEND_API_KEY: 're_test',
  PUBLIC_SITE_URL: 'http://localhost:4321',
};

describe('createAuth', () => {
  it('returns a Better Auth instance with handler + api', () => {
    const auth = createAuth(fakeEnv);
    expect(auth.handler).toBeInstanceOf(Function);
    expect(auth.api).toBeDefined();
  });

  it('throws on missing BETTER_AUTH_SECRET', () => {
    expect(() => createAuth({ ...fakeEnv, BETTER_AUTH_SECRET: '' })).toThrow(/secret/i);
  });
});
```

**Step 2: Implement.**

```ts
// auth.ts
import { betterAuth } from 'better-auth';
import { drizzleAdapter } from 'better-auth/adapters/drizzle';
import { magicLink } from 'better-auth/plugins';
import { getDb } from './db';
import { sendMagicLinkEmail } from './email';

export interface AuthEnv {
  DB: D1Database;
  BETTER_AUTH_SECRET: string;
  GOOGLE_CLIENT_ID: string;
  GOOGLE_CLIENT_SECRET: string;
  GITHUB_CLIENT_ID: string;
  GITHUB_CLIENT_SECRET: string;
  RESEND_API_KEY: string;
  PUBLIC_SITE_URL: string;
}

export function createAuth(env: AuthEnv) {
  if (!env.BETTER_AUTH_SECRET) throw new Error('BETTER_AUTH_SECRET is required');
  const db = getDb(env.DB);
  return betterAuth({
    database: drizzleAdapter(db, { provider: 'sqlite' }),
    secret: env.BETTER_AUTH_SECRET,
    baseURL: env.PUBLIC_SITE_URL,
    socialProviders: {
      google: { clientId: env.GOOGLE_CLIENT_ID, clientSecret: env.GOOGLE_CLIENT_SECRET },
      github: { clientId: env.GITHUB_CLIENT_ID, clientSecret: env.GITHUB_CLIENT_SECRET },
    },
    plugins: [
      magicLink({
        sendMagicLink: async ({ email, url }) => {
          await sendMagicLinkEmail(env.RESEND_API_KEY, email, url);
        },
      }),
    ],
    advanced: {
      cookiePrefix: '__Secure-tinker',
      useSecureCookies: env.PUBLIC_SITE_URL.startsWith('https://'),
    },
    session: {
      expiresIn: 60 * 60 * 24 * 30, // 30 days
      updateAge: 60 * 60 * 24,       // refresh once per day
    },
  });
}
```

**Step 3: Run tests.**

Run: `pnpm test src/server/auth.test.ts`
Expected: 2 passed (after `email.ts` stub from next task — temporarily stub `sendMagicLinkEmail` if needed).

**Step 4: Commit.**

```bash
git add apps/docs/src/server/auth.ts apps/docs/src/server/auth.test.ts
git commit -m "feat(auth): Better Auth config with D1 + magic link + Google + GitHub"
```

---

## Task 2.2: Email sender

**Files:**
- Create: `apps/docs/src/server/email.ts`
- Test: `apps/docs/src/server/email.test.ts`
- Create: `apps/docs/src/server/email-templates.ts`

**Step 1: Write the failing test.**

```ts
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { sendMagicLinkEmail, signUnsubscribeToken, verifyUnsubscribeToken } from './email';

global.fetch = vi.fn(async () => new Response(JSON.stringify({ id: 'msg_123' }), { status: 200 }));

beforeEach(() => { vi.mocked(global.fetch).mockClear(); });

describe('sendMagicLinkEmail', () => {
  it('POSTs to Resend with correct shape', async () => {
    await sendMagicLinkEmail('re_test', 'a@b.co', 'https://learntinker.com/verify?token=x');
    expect(global.fetch).toHaveBeenCalledWith(
      'https://api.resend.com/emails',
      expect.objectContaining({ method: 'POST' })
    );
    const body = JSON.parse((vi.mocked(global.fetch).mock.calls[0][1] as RequestInit).body as string);
    expect(body.to).toEqual(['a@b.co']);
    expect(body.html).toContain('https://learntinker.com/verify?token=x');
  });
});

describe('unsubscribe token', () => {
  it('round-trips a valid token', async () => {
    const tok = await signUnsubscribeToken('user-123', 'secret');
    expect(await verifyUnsubscribeToken(tok, 'secret')).toBe('user-123');
  });
  it('rejects a token signed with a different secret', async () => {
    const tok = await signUnsubscribeToken('user-123', 'secret');
    expect(await verifyUnsubscribeToken(tok, 'other')).toBeNull();
  });
});
```

**Step 2: Implement.** Use Web Crypto's `HMAC-SHA256` (works in Workers).

```ts
// email.ts
import { magicLinkTemplate, welcomeTemplate, dropTemplate } from './email-templates';

const RESEND = 'https://api.resend.com/emails';
const FROM = 'Tinker <hello@learntinker.com>';

export async function sendMagicLinkEmail(apiKey: string, to: string, url: string) {
  await postResend(apiKey, {
    from: FROM,
    to: [to],
    subject: 'Sign in to Tinker',
    html: magicLinkTemplate({ url }),
  });
}

export async function sendWelcomeEmail(apiKey: string, to: string) {
  await postResend(apiKey, {
    from: FROM,
    to: [to],
    subject: 'Welcome to Tinker',
    html: welcomeTemplate(),
  });
}

export async function sendDropEmail(apiKey: string, to: string, subject: string, bodyHtml: string, unsubUrl: string) {
  await postResend(apiKey, {
    from: FROM,
    to: [to],
    subject,
    html: dropTemplate({ bodyHtml, unsubUrl }),
    headers: { 'List-Unsubscribe': `<${unsubUrl}>`, 'List-Unsubscribe-Post': 'List-Unsubscribe=One-Click' },
  });
}

async function postResend(apiKey: string, payload: unknown) {
  const res = await fetch(RESEND, {
    method: 'POST',
    headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`Resend ${res.status}: ${await res.text()}`);
  return res.json();
}

// HMAC-signed unsubscribe tokens (no DB roundtrip on click)
export async function signUnsubscribeToken(userId: string, secret: string): Promise<string> {
  const key = await importKey(secret);
  const sig = await crypto.subtle.sign('HMAC', key, new TextEncoder().encode(userId));
  return `${userId}.${b64u(sig)}`;
}

export async function verifyUnsubscribeToken(token: string, secret: string): Promise<string | null> {
  const [userId, sig] = token.split('.');
  if (!userId || !sig) return null;
  const key = await importKey(secret);
  const ok = await crypto.subtle.verify('HMAC', key, b64uDecode(sig), new TextEncoder().encode(userId));
  return ok ? userId : null;
}

async function importKey(secret: string) {
  return crypto.subtle.importKey('raw', new TextEncoder().encode(secret), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign', 'verify']);
}
function b64u(buf: ArrayBuffer) { return btoa(String.fromCharCode(...new Uint8Array(buf))).replace(/\+/g,'-').replace(/\//g,'_').replace(/=/g,''); }
function b64uDecode(s: string) { const b = atob(s.replace(/-/g,'+').replace(/_/g,'/')); return Uint8Array.from(b, c => c.charCodeAt(0)).buffer; }
```

Templates in `email-templates.ts` are plain functions returning HTML strings (no React Email runtime needed for v1; can swap later):

```ts
export const magicLinkTemplate = ({ url }: { url: string }) => `
<!doctype html><html><body style="font-family:system-ui;max-width:560px;margin:auto;padding:24px">
<h2>Sign in to Tinker</h2>
<p>Click below to sign in. This link expires in 15 minutes.</p>
<p><a href="${url}" style="display:inline-block;padding:12px 20px;background:#000;color:#fff;border-radius:8px;text-decoration:none">Sign in</a></p>
<p style="color:#666;font-size:14px">If you didn't request this, ignore this email.</p>
</body></html>`;

export const welcomeTemplate = () => `
<!doctype html><html><body style="font-family:system-ui;max-width:560px;margin:auto;padding:24px">
<h2>You're in.</h2>
<p>Tinker is in alpha. One course is live today: <a href="https://learntinker.com/courses/ml-math">Math for Machine Learning</a>. New modules every week.</p>
<p>You can opt into a heads-up email when new modules ship at <a href="https://learntinker.com/me">your account page</a>.</p>
</body></html>`;

export const dropTemplate = ({ bodyHtml, unsubUrl }: { bodyHtml: string; unsubUrl: string }) => `
<!doctype html><html><body style="font-family:system-ui;max-width:560px;margin:auto;padding:24px">
${bodyHtml}
<hr style="margin:32px 0;border:none;border-top:1px solid #ddd">
<p style="color:#888;font-size:12px"><a href="${unsubUrl}" style="color:#888">Unsubscribe</a></p>
</body></html>`;
```

**Step 3: Run tests.** Expected: 4 passed.

**Step 4: Commit.**

```bash
git add apps/docs/src/server/email.ts apps/docs/src/server/email-templates.ts apps/docs/src/server/email.test.ts
git commit -m "feat(email): Resend wrapper, templates, HMAC unsubscribe tokens"
```

---

## Task 2.3: Rate limiter

**Files:**
- Create: `apps/docs/src/server/ratelimit.ts`
- Test: `apps/docs/src/server/ratelimit.test.ts`
- Modify: `apps/docs/src/server/schema.ts` (add `rate_limit` table)
- Modify: `apps/docs/migrations/` (regenerate)

**Step 1: Add `rate_limit` table to schema.**

```ts
export const rateLimit = sqliteTable('rate_limit', {
  key: text('key').primaryKey(),     // e.g. "magic-link:email:foo@bar.co"
  count: integer('count').notNull(),
  resetAt: integer('reset_at', { mode: 'timestamp' }).notNull(),
});
```

Regenerate migration: `pnpm drizzle-kit generate` and apply locally.

**Step 2: Write the failing test.**

```ts
import { describe, it, expect, beforeEach } from 'vitest';
import { checkRateLimit } from './ratelimit';
import { makeTestDb } from '../../tests/support/d1';

let db: ReturnType<typeof makeTestDb>;
beforeEach(() => { db = makeTestDb(); });

describe('checkRateLimit', () => {
  it('allows up to limit then blocks', async () => {
    for (let i = 0; i < 3; i++) {
      const r = await checkRateLimit(db.client, 'k', { limit: 3, windowMs: 60_000 });
      expect(r.allowed).toBe(true);
    }
    const r = await checkRateLimit(db.client, 'k', { limit: 3, windowMs: 60_000 });
    expect(r.allowed).toBe(false);
    expect(r.retryAfterMs).toBeGreaterThan(0);
  });

  it('resets after window expires', async () => {
    await checkRateLimit(db.client, 'k', { limit: 1, windowMs: 1 });
    await new Promise(r => setTimeout(r, 5));
    const r = await checkRateLimit(db.client, 'k', { limit: 1, windowMs: 1 });
    expect(r.allowed).toBe(true);
  });
});
```

**Step 3: Implement.** Sliding window with single-row upsert.

```ts
// ratelimit.ts
import { eq } from 'drizzle-orm';
import { rateLimit } from './schema';
import type { DB } from './db';

export async function checkRateLimit(db: DB, key: string, opts: { limit: number; windowMs: number }) {
  const now = new Date();
  const row = await db.select().from(rateLimit).where(eq(rateLimit.key, key)).get();

  if (!row || row.resetAt.getTime() < now.getTime()) {
    await db.insert(rateLimit)
      .values({ key, count: 1, resetAt: new Date(now.getTime() + opts.windowMs) })
      .onConflictDoUpdate({ target: rateLimit.key, set: { count: 1, resetAt: new Date(now.getTime() + opts.windowMs) } });
    return { allowed: true, remaining: opts.limit - 1, retryAfterMs: 0 };
  }

  if (row.count >= opts.limit) {
    return { allowed: false, remaining: 0, retryAfterMs: row.resetAt.getTime() - now.getTime() };
  }

  await db.update(rateLimit).set({ count: row.count + 1 }).where(eq(rateLimit.key, key));
  return { allowed: true, remaining: opts.limit - 1 - row.count, retryAfterMs: 0 };
}
```

**Step 4: Run tests.** Expected: 2 passed.

**Step 5: Commit.**

```bash
git add apps/docs/src/server/ratelimit.ts apps/docs/src/server/ratelimit.test.ts apps/docs/src/server/schema.ts apps/docs/migrations/
git commit -m "feat(auth): D1-backed sliding-window rate limiter"
```

---

## Task 2.4: Test DB helper for integration tests

**Files:**
- Create: `apps/docs/tests/support/d1.ts`
- Add dev dep: `better-sqlite3`

**Step 1: Install.**

```bash
cd apps/docs && pnpm add -D better-sqlite3 @types/better-sqlite3
```

**Step 2: Implement helper.** Apply migrations by splitting the SQL on statement boundaries and running each via `prepare().run()`:

```ts
// d1.ts
import Database from 'better-sqlite3';
import { drizzle } from 'drizzle-orm/better-sqlite3';
import * as schema from '../../src/server/schema';
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';

export function makeTestDb() {
  const sqlite = new Database(':memory:');
  const client = drizzle(sqlite, { schema });
  const dir = join(__dirname, '../../migrations');
  const files = readdirSync(dir).filter(f => f.endsWith('.sql')).sort();
  for (const f of files) {
    const sql = readFileSync(join(dir, f), 'utf8');
    for (const stmt of sql.split('--> statement-breakpoint').map(s => s.trim()).filter(Boolean)) {
      sqlite.prepare(stmt).run();
    }
  }
  return { client, sqlite, close: () => sqlite.close() };
}
```

> Drizzle generates SQL files with `--> statement-breakpoint` separators. Split on those, run each prepared statement.

**Step 3: Verify by re-running ratelimit tests.** Expected: still 2 passed.

**Step 4: Commit.**

```bash
git add apps/docs/tests/support/d1.ts apps/docs/package.json
git commit -m "test: in-memory D1 helper for integration tests"
```

---

## Task 2.5: Mount Better Auth handler

**Files:**
- Create: `apps/docs/src/pages/api/auth/[...all].ts`
- Create: `apps/docs/src/env.d.ts` (or modify existing)

**Step 1: Define `Astro.locals` types.** In `src/env.d.ts`:

```ts
/// <reference types="astro/client" />
type Runtime = import('@astrojs/cloudflare').Runtime<Env>;
declare namespace App {
  interface Locals extends Runtime {
    user?: import('better-auth').User;
    session?: import('better-auth').Session;
  }
}
```

`Env` is auto-generated by `wrangler types` — run `pnpm generate-types` after Phase 1 to ensure it's up to date.

**Step 2: Mount Better Auth handler.**

```ts
// [...all].ts
import type { APIRoute } from 'astro';
import { createAuth } from '../../../server/auth';

export const prerender = false;

export const ALL: APIRoute = async ({ request, locals }) => {
  const env = (locals as any).runtime.env as ImportMetaEnv & Record<string, string> & { DB: D1Database };
  const auth = createAuth({
    DB: env.DB,
    BETTER_AUTH_SECRET: env.BETTER_AUTH_SECRET,
    GOOGLE_CLIENT_ID: env.GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET: env.GOOGLE_CLIENT_SECRET,
    GITHUB_CLIENT_ID: env.GITHUB_CLIENT_ID,
    GITHUB_CLIENT_SECRET: env.GITHUB_CLIENT_SECRET,
    RESEND_API_KEY: env.RESEND_API_KEY,
    PUBLIC_SITE_URL: env.PUBLIC_SITE_URL ?? new URL(request.url).origin,
  });
  return auth.handler(request);
};
```

**Step 3: Manual smoke.**

Run: `cd apps/docs && pnpm dev`
Then: `curl -i http://localhost:4321/api/auth/sign-in/magic-link -X POST -H 'content-type: application/json' -d '{"email":"test@example.com"}'`
Expected: 200 (or whatever Better Auth returns); a row appears in local D1's `verification` table.

Verify: `pnpm dlx wrangler@latest d1 execute tinker --local --command "select * from verification"` — should show the token.

**Step 4: Commit.**

```bash
git add apps/docs/src/pages/api/auth apps/docs/src/env.d.ts
git commit -m "feat(auth): mount Better Auth handler at /api/auth/[...all]"
```

---

## Task 2.6: Auth integration tests

**Files:**
- Create: `apps/docs/tests/integration/auth.test.ts`

**Step 1: Test the magic link flow end-to-end against in-memory D1.**

```ts
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { makeTestDb } from '../support/d1';
import { betterAuth } from 'better-auth';
import { drizzleAdapter } from 'better-auth/adapters/drizzle';
import { magicLink } from 'better-auth/plugins';

let sentLinks: { email: string; url: string }[] = [];
let db: ReturnType<typeof makeTestDb>;
let auth: ReturnType<typeof betterAuth>;

beforeEach(() => {
  sentLinks = [];
  db = makeTestDb();
  auth = betterAuth({
    database: drizzleAdapter(db.client, { provider: 'sqlite' }),
    secret: 'a'.repeat(32),
    baseURL: 'http://localhost',
    plugins: [magicLink({ sendMagicLink: async ({ email, url }) => { sentLinks.push({ email, url }); } })],
  });
});

describe('magic link flow', () => {
  it('creates a verification row on request', async () => {
    const res = await auth.handler(new Request('http://localhost/api/auth/sign-in/magic-link', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ email: 'a@b.co' }),
    }));
    expect(res.status).toBeLessThan(400);
    expect(sentLinks).toHaveLength(1);
    expect(sentLinks[0].email).toBe('a@b.co');
  });

  it('verifies and creates session on click', async () => {
    await auth.handler(new Request('http://localhost/api/auth/sign-in/magic-link', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ email: 'a@b.co' }),
    }));
    const verifyRes = await auth.handler(new Request(sentLinks[0].url));
    expect(verifyRes.status).toBeLessThan(400);
    const { session } = await import('../../src/server/schema');
    const rows = await db.client.select().from(session).all();
    expect(rows.length).toBeGreaterThan(0);
  });

  it('rejects re-used token', async () => {
    await auth.handler(new Request('http://localhost/api/auth/sign-in/magic-link', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ email: 'a@b.co' }),
    }));
    await auth.handler(new Request(sentLinks[0].url));
    const second = await auth.handler(new Request(sentLinks[0].url));
    expect(second.status).toBeGreaterThanOrEqual(400);
  });

  it('rejects expired token', async () => {
    vi.useFakeTimers();
    await auth.handler(new Request('http://localhost/api/auth/sign-in/magic-link', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ email: 'a@b.co' }),
    }));
    vi.advanceTimersByTime(16 * 60 * 1000);
    const res = await auth.handler(new Request(sentLinks[0].url));
    expect(res.status).toBeGreaterThanOrEqual(400);
    vi.useRealTimers();
  });
});
```

**Step 2: Run.**

Run: `pnpm test tests/integration/auth.test.ts`
Expected: 4 passed.

**Step 3: Commit.**

```bash
git add apps/docs/tests/integration/auth.test.ts
git commit -m "test(auth): integration tests for magic link request, verify, replay, expiry"
```

---

# Phase 3 — Auth UI

## Task 3.1: Sign-in / sign-up page

**Files:**
- Create: `apps/docs/src/pages/signin.astro`
- Create: `apps/docs/src/pages/signup.astro` (re-export of signin with copy variant)
- Create: `apps/docs/src/components/auth/AuthForm.svelte`

**Step 1: Write the AuthForm component.** Three OAuth buttons + email field with Turnstile widget. On submit POSTs to `/api/auth/sign-in/magic-link`. On success shows "Check your email."

```svelte
<!-- AuthForm.svelte -->
<script lang="ts">
  let { mode = 'signin', turnstileSiteKey = '' } = $props<{ mode?: 'signin' | 'signup'; turnstileSiteKey?: string }>();
  let email = $state('');
  let status = $state<'idle'|'sending'|'sent'|'error'>('idle');
  let errorMsg = $state('');

  async function submitEmail(e: SubmitEvent) {
    e.preventDefault();
    status = 'sending';
    try {
      const res = await fetch('/api/auth/sign-in/magic-link', {
        method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ email, callbackURL: '/welcome' }),
      });
      if (!res.ok) throw new Error(await res.text());
      status = 'sent';
    } catch (e: any) {
      status = 'error';
      errorMsg = e.message || 'Something went wrong.';
    }
  }
</script>

<div class="auth-card">
  <h1>{mode === 'signup' ? 'Sign up for Tinker' : 'Sign in to Tinker'}</h1>
  <a class="btn provider" href="/api/auth/sign-in/google">Continue with Google</a>
  <a class="btn provider" href="/api/auth/sign-in/github">Continue with GitHub</a>
  <div class="sep">or</div>
  {#if status === 'sent'}
    <p>Check your inbox for a sign-in link.</p>
  {:else}
    <form onsubmit={submitEmail}>
      <input type="email" bind:value={email} required placeholder="you@example.com" autocomplete="email" />
      {#if turnstileSiteKey}
        <div class="cf-turnstile" data-sitekey={turnstileSiteKey} data-size="invisible"></div>
      {/if}
      <button type="submit" disabled={status === 'sending'}>
        {status === 'sending' ? 'Sending…' : 'Send sign-in link'}
      </button>
      {#if status === 'error'}<p class="err">{errorMsg}</p>{/if}
    </form>
  {/if}
</div>

<style>
  /* Use design tokens — read DESIGN.md before adding any color or spacing */
  .auth-card { max-width: 420px; margin: 4rem auto; padding: 2rem; }
  .provider { display: block; padding: 0.75rem; text-align: center; border: 1px solid var(--color-border); border-radius: 8px; margin-bottom: 0.5rem; }
  .sep { text-align: center; color: var(--color-fg-muted); margin: 1rem 0; }
  input { width: 100%; padding: 0.75rem; border: 1px solid var(--color-border); border-radius: 8px; }
  button { width: 100%; padding: 0.75rem; background: var(--color-fg); color: var(--color-bg); border-radius: 8px; margin-top: 0.5rem; }
  .err { color: var(--color-danger); }
</style>
```

> ⚠️ **Read `DESIGN.md` first.** All color/spacing/font tokens come from there + `apps/docs/src/styles/global.css`. The placeholders above (`var(--color-border)` etc.) must match real tokens. Replace any that don't exist with the actual ones, or add them to DESIGN.md first.

**Step 2: Wire pages.** `apps/docs/src/pages/signin.astro`:

```astro
---
import Base from '../layouts/Base.astro';
import AuthForm from '../components/auth/AuthForm.svelte';
const turnstileSiteKey = (Astro.locals as any).runtime.env.TURNSTILE_SITE_KEY ?? '';
---
<Base title="Sign in — Tinker">
  <AuthForm client:load mode="signin" {turnstileSiteKey} />
  <script src="https://challenges.cloudflare.com/turnstile/v0/api.js" async defer></script>
</Base>
```

`signup.astro` is identical with `mode="signup"`.

**Step 3: Manual smoke.** `pnpm dev`, visit `/signin`, type your email, check inbox (Resend dashboard if not yet verified, or `--local` D1 verification table for dev).

**Step 4: Commit.**

```bash
git add apps/docs/src/pages/signin.astro apps/docs/src/pages/signup.astro apps/docs/src/components/auth/AuthForm.svelte
git commit -m "feat(auth): sign-in and sign-up pages with magic link + OAuth + Turnstile"
```

---

## Task 3.2: Verify / welcome / signout pages

**Files:**
- Create: `apps/docs/src/pages/verify.astro`
- Create: `apps/docs/src/pages/welcome.astro`

**Step 1: Verify page.** `verify.astro` is server-rendered. It detects the verification status from query params (Better Auth's redirect target with `?error=…` or success). On error, show "expired or already used — request a new link." On success, redirect to `/welcome` for first-time users or `/` for returning.

```astro
---
export const prerender = false;
import Base from '../layouts/Base.astro';
const url = new URL(Astro.request.url);
const error = url.searchParams.get('error');
---
<Base title="Verifying — Tinker">
  {error ? (
    <div class="auth-card">
      <h1>Link expired or already used</h1>
      <p>Sign-in links are single-use and last 15 minutes. Request a new one.</p>
      <a href="/signin">Send a new link</a>
    </div>
  ) : (
    <div class="auth-card">
      <h1>Signing you in…</h1>
      <script>window.location.replace('/welcome');</script>
    </div>
  )}
</Base>
```

**Step 2: Welcome page.** First-time signup: insert `user_profile` row, show marketing opt-in checkbox.

```astro
---
export const prerender = false;
import Base from '../layouts/Base.astro';
import { createAuth } from '../server/auth';
import { getDb } from '../server/db';
import { userProfile } from '../server/schema';
import { eq } from 'drizzle-orm';

const env = (Astro.locals as any).runtime.env;
const auth = createAuth(env);
const session = await auth.api.getSession({ headers: Astro.request.headers });
if (!session) return Astro.redirect('/signin');

const db = getDb(env.DB);
const existing = await db.select().from(userProfile).where(eq(userProfile.userId, session.user.id)).get();
if (!existing) {
  await db.insert(userProfile).values({ userId: session.user.id, marketingOptIn: false });
}

if (Astro.request.method === 'POST') {
  const form = await Astro.request.formData();
  const optIn = form.get('marketing_opt_in') === 'on';
  await db.update(userProfile)
    .set({ marketingOptIn: optIn, onboardedAt: new Date() })
    .where(eq(userProfile.userId, session.user.id));
  return Astro.redirect('/courses/ml-math');
}
---
<Base title="Welcome — Tinker">
  <div class="auth-card">
    <h1>You're in.</h1>
    <p>Tinker is in alpha. One course is live today: Math for Machine Learning. New modules every week.</p>
    <form method="POST">
      <label><input type="checkbox" name="marketing_opt_in" /> Email me when new modules drop.</label>
      <button type="submit">Start learning</button>
    </form>
  </div>
</Base>
```

**Step 3: Verify sign-out works.** Better Auth handles `POST /api/auth/sign-out`. Add a sign-out button in `<UserMenu>` (next task) that POSTs there.

**Step 4: Commit.**

```bash
git add apps/docs/src/pages/verify.astro apps/docs/src/pages/welcome.astro
git commit -m "feat(auth): verify and welcome pages with first-time profile creation"
```

---

## Task 3.3: User menu in nav

**Files:**
- Create: `apps/docs/src/components/auth/UserMenu.svelte`
- Modify: `apps/docs/src/components/Nav.astro`

**Step 1: UserMenu.** Logged-out: "Sign in" + "Sign up" links. Logged-in: avatar + dropdown (Dashboard, Bookmarks, Sign out).

```svelte
<!-- UserMenu.svelte -->
<script lang="ts">
  let { user } = $props<{ user: { id: string; email: string; name?: string | null; image?: string | null } | null }>();
  let open = $state(false);
  async function signOut() {
    await fetch('/api/auth/sign-out', { method: 'POST' });
    location.href = '/';
  }
</script>

{#if user}
  <div class="usermenu">
    <button onclick={() => open = !open} aria-haspopup="menu" aria-expanded={open}>
      {#if user.image}<img src={user.image} alt="" />{:else}<span class="initial">{(user.name ?? user.email)[0].toUpperCase()}</span>{/if}
    </button>
    {#if open}
      <ul role="menu">
        <li><a href="/me">Dashboard</a></li>
        <li><a href="/me/bookmarks">Bookmarks</a></li>
        <li><button onclick={signOut}>Sign out</button></li>
      </ul>
    {/if}
  </div>
{:else}
  <a href="/signin">Sign in</a>
  <a class="btn" href="/signup">Sign up</a>
{/if}
```

**Step 2: Wire into Nav.astro.** Read session server-side, pass to component.

```astro
---
import { createAuth } from '../server/auth';
import UserMenu from './auth/UserMenu.svelte';
const env = (Astro.locals as any).runtime.env;
const auth = createAuth(env);
const session = env?.DB ? await auth.api.getSession({ headers: Astro.request.headers }) : null;
---
<!-- existing nav markup -->
<UserMenu user={session?.user ?? null} client:load />
```

**Step 3: Manual smoke.** Visit `/`, see "Sign in / Sign up". Sign in via magic link. Refresh: see avatar.

**Step 4: Commit.**

```bash
git add apps/docs/src/components/auth/UserMenu.svelte apps/docs/src/components/Nav.astro
git commit -m "feat(auth): UserMenu in nav with sign-in / sign-up / dropdown"
```

---

## Task 3.4: /me dashboard page

**Files:**
- Create: `apps/docs/src/pages/me.astro`
- Create: `apps/docs/src/pages/api/me/index.ts` (DELETE handler)

**Step 1: /me.astro.** Shows: email, marketing opt-in toggle, "Delete my account" button. Account-scoped only — bookmarks and progress get separate pages.

**Step 2: DELETE /api/me.** Cascades + anonymizes email + sends deletion email + invalidates session.

```ts
// /api/me/index.ts
import type { APIRoute } from 'astro';
import { eq } from 'drizzle-orm';
import { createAuth } from '../../../server/auth';
import { getDb } from '../../../server/db';
import { user } from '../../../server/schema';
export const prerender = false;

export const DELETE: APIRoute = async ({ request, locals }) => {
  const env = (locals as any).runtime.env;
  const auth = createAuth(env);
  const session = await auth.api.getSession({ headers: request.headers });
  if (!session) return new Response('unauthorized', { status: 401 });
  const db = getDb(env.DB);
  const placeholder = `deleted-${crypto.randomUUID()}@deleted.local`;
  await db.update(user).set({ email: placeholder, name: null, image: null }).where(eq(user.id, session.user.id));
  await auth.api.signOut({ headers: request.headers });
  return new Response(null, { status: 204 });
};
```

**Step 3: Test (integration).** `tests/integration/me.test.ts`: signed-in user → DELETE → user email is anonymized → session destroyed.

**Step 4: Commit.**

```bash
git add apps/docs/src/pages/me.astro apps/docs/src/pages/api/me apps/docs/tests/integration/me.test.ts
git commit -m "feat(account): /me dashboard with marketing toggle + account deletion"
```

---

# Phase 4 — Progress

## Task 4.1: Progress endpoints

**Files:**
- Create: `apps/docs/src/pages/api/progress/view.ts`
- Create: `apps/docs/src/pages/api/progress/complete.ts`
- Create: `apps/docs/src/pages/api/progress/merge.ts`
- Create: `apps/docs/src/server/middleware.ts` (`requireSession`, CSRF check, rate limit wrapper)
- Test: `apps/docs/tests/integration/progress.test.ts`

**Step 1: Middleware helper.**

```ts
// middleware.ts
import { createAuth } from './auth';
import { getDb } from './db';

export async function requireSession(request: Request, env: any) {
  const auth = createAuth(env);
  const session = await auth.api.getSession({ headers: request.headers });
  if (!session) return { error: new Response('unauthorized', { status: 401 }) } as const;
  return { session, db: getDb(env.DB), env } as const;
}

export function requireCsrf(request: Request) {
  const cookieHeader = request.headers.get('cookie') ?? '';
  const csrfCookie = /(?:^|;\s*)__Secure-tinker\.csrf=([^;]+)/.exec(cookieHeader)?.[1];
  const csrfHeader = request.headers.get('x-tinker-csrf');
  if (!csrfCookie || csrfCookie !== csrfHeader) return new Response('forbidden', { status: 403 });
  return null;
}
```

> Note: Better Auth sets a CSRF cookie automatically. Confirm the cookie name in their docs and update `requireCsrf` to match if different.

**Step 2: `view` endpoint.**

```ts
// view.ts
import type { APIRoute } from 'astro';
import { z } from 'zod';
import { lessonView } from '../../../server/schema';
import { sql } from 'drizzle-orm';
import { requireSession, requireCsrf } from '../../../server/middleware';
import { checkRateLimit } from '../../../server/ratelimit';
export const prerender = false;

const Body = z.object({ lesson_slug: z.string().min(1).max(200), course_slug: z.string().min(1).max(100), module_slug: z.string().min(1).max(100) });

export const POST: APIRoute = async ({ request, locals }) => {
  const csrf = requireCsrf(request); if (csrf) return csrf;
  const ctx = await requireSession(request, (locals as any).runtime.env);
  if ('error' in ctx) return ctx.error;
  const rl = await checkRateLimit(ctx.db, `progress:${ctx.session.user.id}`, { limit: 60, windowMs: 60_000 });
  if (!rl.allowed) return new Response('rate limited', { status: 429 });

  const parsed = Body.safeParse(await request.json());
  if (!parsed.success) return new Response('bad request', { status: 400 });

  const now = new Date();
  await ctx.db.insert(lessonView)
    .values({
      userId: ctx.session.user.id,
      courseSlug: parsed.data.course_slug,
      moduleSlug: parsed.data.module_slug,
      lessonSlug: parsed.data.lesson_slug,
      firstSeenAt: now,
      lastSeenAt: now,
      viewCount: 1,
    })
    .onConflictDoUpdate({
      target: [lessonView.userId, lessonView.lessonSlug],
      set: { lastSeenAt: now, viewCount: sql`${lessonView.viewCount} + 1` },
    });

  return new Response(null, { status: 204 });
};
```

**Step 3: `complete` endpoint.** Same shape, sets `completedAt = now` only if currently null (use `coalesce` or a where clause on conflict).

**Step 4: `merge` endpoint.** Accepts an array of `lessonView` rows from localStorage, idempotently upserts each, never overwrites a `completedAt` that's already set.

**Step 5: Integration tests.**

```ts
// progress.test.ts
// Tests: 401 without session, 200 with session, view_count increments, completedAt only moves forward, merge respects existing completedAt
```

**Step 6: Run.** Expected: 6 passed.

**Step 7: Commit.**

```bash
git add apps/docs/src/pages/api/progress apps/docs/src/server/middleware.ts apps/docs/tests/integration/progress.test.ts
git commit -m "feat(progress): view, complete, merge endpoints with rate limit + CSRF"
```

---

## Task 4.2: ProgressBeacon component

**Files:**
- Create: `apps/docs/src/components/lesson/ProgressBeacon.svelte`

**Step 1: Implement.** On mount: read CSRF from cookie, fire view POST (debounced 2s — covers refreshes within 2s). On `visibilitychange === hidden`: fire view POST again. For anonymous users: write to localStorage `tinker:progress:v1` with same shape.

```svelte
<script lang="ts">
  let { lessonSlug, courseSlug, moduleSlug, isAuthed } = $props<{
    lessonSlug: string; courseSlug: string; moduleSlug: string; isAuthed: boolean;
  }>();

  function getCsrf(): string {
    const m = /(?:^|;\s*)__Secure-tinker\.csrf=([^;]+)/.exec(document.cookie);
    return m?.[1] ?? '';
  }

  async function postView() {
    if (isAuthed) {
      await fetch('/api/progress/view', {
        method: 'POST',
        headers: { 'content-type': 'application/json', 'x-tinker-csrf': getCsrf() },
        body: JSON.stringify({ lesson_slug: lessonSlug, course_slug: courseSlug, module_slug: moduleSlug }),
        keepalive: true,
      });
    } else {
      const key = 'tinker:progress:v1';
      const arr = JSON.parse(localStorage.getItem(key) ?? '[]');
      const existing = arr.find((e: any) => e.lesson_slug === lessonSlug);
      const now = new Date().toISOString();
      if (existing) { existing.last_seen_at = now; existing.view_count = (existing.view_count ?? 1) + 1; }
      else arr.push({ lesson_slug: lessonSlug, course_slug: courseSlug, module_slug: moduleSlug, first_seen_at: now, last_seen_at: now, view_count: 1 });
      localStorage.setItem(key, JSON.stringify(arr));
    }
  }

  let timer: number | undefined;
  $effect(() => {
    timer = window.setTimeout(postView, 2000);
    const onHide = () => { if (document.visibilityState === 'hidden') postView(); };
    document.addEventListener('visibilitychange', onHide);
    return () => { clearTimeout(timer); document.removeEventListener('visibilitychange', onHide); };
  });
</script>
```

**Step 2: Mount it on lesson pages.** In `apps/docs/src/layouts/Lesson.astro` (the layout used by lesson MDX), add the beacon at the top of the body:

```astro
---
import ProgressBeacon from '../components/lesson/ProgressBeacon.svelte';
import { createAuth } from '../server/auth';
const { lessonSlug, courseSlug, moduleSlug } = Astro.props.frontmatter;
const env = (Astro.locals as any).runtime.env;
const auth = createAuth(env);
const session = await auth.api.getSession({ headers: Astro.request.headers });
---
<ProgressBeacon client:load {lessonSlug} {courseSlug} {moduleSlug} isAuthed={!!session} />
<slot />
```

> Note: Lessons must expose `lessonSlug`, `courseSlug`, `moduleSlug` in their MDX frontmatter. This may already be the case — confirm by reading `apps/docs/src/content.config.ts` and a lesson MDX file. If not, this is a one-time content migration as part of this task.

**Step 3: Smoke test.** Sign in, visit a lesson, check D1 `lesson_view` row appears.

**Step 4: Commit.**

```bash
git add apps/docs/src/components/lesson/ProgressBeacon.svelte apps/docs/src/layouts/Lesson.astro
git commit -m "feat(progress): ProgressBeacon mounted on every lesson page"
```

---

## Task 4.3: "Continue where you left off" + per-module checkmarks

**Files:**
- Modify: `apps/docs/src/pages/courses/[course]/index.astro`

**Step 1: Server-render the user's most-recent + completion map.**

```astro
---
export const prerender = false;
import { createAuth } from '../../../server/auth';
import { getDb } from '../../../server/db';
import { lessonView, user as userTbl } from '../../../server/schema';
import { eq, and, desc } from 'drizzle-orm';

const env = (Astro.locals as any).runtime.env;
const auth = createAuth(env);
const session = await auth.api.getSession({ headers: Astro.request.headers });
const courseSlug = Astro.params.course!;

let resumeLesson: { slug: string; title?: string } | null = null;
let completed = new Set<string>();
let learnerCount = 0;
if (env?.DB) {
  const db = getDb(env.DB);
  if (session) {
    const recent = await db.select().from(lessonView)
      .where(and(eq(lessonView.userId, session.user.id), eq(lessonView.courseSlug, courseSlug)))
      .orderBy(desc(lessonView.lastSeenAt)).limit(1).get();
    if (recent) resumeLesson = { slug: recent.lessonSlug };
    const all = await db.select({ s: lessonView.lessonSlug, c: lessonView.completedAt }).from(lessonView)
      .where(eq(lessonView.userId, session.user.id));
    for (const r of all) if (r.c) completed.add(r.s);
  }
  learnerCount = await db.$count(userTbl);
}
---
<div class="alpha-banner">🌱 Alpha · 1 course live · new modules weekly · {learnerCount} learners</div>

{resumeLesson && <a class="continue-cta" href={`/courses/${courseSlug}/${resumeLesson.slug}`}>Continue where you left off →</a>}

<!-- Module tree: thread `completed` Set through to add ✓ next to lessons -->
```

**Step 2: Smoke test.** Sign in, visit 3 lessons, return to `/courses/ml-math`, see "Continue where you left off" pointing at the most recent and ✓ next to those three.

**Step 3: Commit.**

```bash
git add apps/docs/src/pages/courses/[course]/index.astro
git commit -m "feat(progress): continue-where-you-left-off CTA + checkmarks + alpha banner"
```

---

## Task 4.4: Anonymous progress merge on first sign-in

**Files:**
- Modify: `apps/docs/src/layouts/Base.astro` — add a one-shot client script when session present

**Step 1: Add a one-shot client script.** When a session exists AND `localStorage.tinker:progress:v1` has entries: POST them to `/api/progress/merge`, clear the key on success.

```astro
<!-- in Base.astro, only when session present -->
{session && <script is:inline>
  (function() {
    const key = 'tinker:progress:v1';
    const raw = localStorage.getItem(key);
    if (!raw) return;
    fetch('/api/progress/merge', {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'x-tinker-csrf': (document.cookie.match(/__Secure-tinker\.csrf=([^;]+)/) || [])[1] || '',
      },
      body: JSON.stringify({ entries: JSON.parse(raw) }),
    }).then(r => { if (r.ok) localStorage.removeItem(key); });
  })();
</script>}
```

**Step 2: Test (integration).** Anon view → sign up → merge endpoint sees the rows → DB has them.

**Step 3: Commit.**

```bash
git add apps/docs/src/layouts/Base.astro
git commit -m "feat(progress): merge anon localStorage progress on first sign-in"
```

---

# Phase 5 — Exercises

## Task 5.1: Exercise endpoint + widget contract

**Files:**
- Create: `apps/docs/src/pages/api/exercises/answer.ts`
- Create: `apps/docs/src/components/lesson/ExerciseShell.svelte` — wraps any widget that opts in
- Test: `apps/docs/tests/integration/exercises.test.ts`

**Step 1: Endpoint.** `POST { lesson_slug, exercise_id, answer_json, is_correct }` → insert row in `exercise_answer`. `attempt_no` = previous max + 1 for `(user_id, lesson_slug, exercise_id)`.

**Step 2: Widget contract.** Widgets that opt in expose `serializeAnswer(): unknown` and `restoreAnswer(json: unknown): void`. `ExerciseShell` receives a slot, an `exerciseId`, and uses Svelte's `bind:this` plus the contract. On submit: `POST /api/exercises/answer`. On mount: GET latest from `/api/me/state` (Task 7.1) and call `restoreAnswer`.

**Step 3: Tests.** Submit → row exists. Submit again → `attempt_no = 2`. Foreign user_id rejected.

**Step 4: Commit.**

```bash
git add apps/docs/src/pages/api/exercises apps/docs/src/components/lesson/ExerciseShell.svelte apps/docs/tests/integration/exercises.test.ts
git commit -m "feat(exercises): persist answer attempts with monotonic attempt_no"
```

---

# Phase 6 — Bookmarks

## Task 6.1: Bookmark endpoints + button + listing

**Files:**
- Create: `apps/docs/src/pages/api/bookmarks/index.ts` (POST)
- Create: `apps/docs/src/pages/api/bookmarks/[id].ts` (DELETE)
- Create: `apps/docs/src/components/lesson/BookmarkButton.svelte`
- Create: `apps/docs/src/pages/me/bookmarks.astro`
- Test: `apps/docs/tests/integration/bookmarks.test.ts`

**Step 1: Endpoints.** Same pattern as progress: requireSession + requireCsrf + Zod + rate limit. POST inserts (handles unique-constraint as 200 noop). DELETE checks ownership before deleting.

**Step 2: BookmarkButton.** Toggle state. On click → POST or DELETE. Reads initial state from `/api/me/state`.

**Step 3: /me/bookmarks page.** SSR query: list user's bookmarks ordered by `createdAt desc`. Each row links to lesson + has a "Remove" button.

**Step 4: Tests.** POST creates, duplicate POST is idempotent, DELETE works, foreign ownership 403.

**Step 5: Commit.**

```bash
git add apps/docs/src/pages/api/bookmarks apps/docs/src/components/lesson/BookmarkButton.svelte apps/docs/src/pages/me/bookmarks.astro apps/docs/tests/integration/bookmarks.test.ts
git commit -m "feat(bookmarks): create, list, remove + lesson button + /me/bookmarks page"
```

---

# Phase 7 — Notes + hydration endpoint

## Task 7.1: GET /api/me/state (single hydration call)

**Files:**
- Create: `apps/docs/src/pages/api/me/state.ts`
- Test: `apps/docs/tests/integration/me-state.test.ts`

**Step 1: Implement.** Query for current user: progress map (lesson_slug → {viewCount, completedAt, lastSeenAt}), exercise latest-attempt map, bookmark list, notes map (lesson_slug → updatedAt only — body fetched per-lesson).

```ts
// state.ts — returns { progress, exercises, bookmarks, notes_index }
```

**Step 2: Test.** With seeded DB, returns expected shape; foreign user data not leaked.

**Step 3: Commit.**

```bash
git add apps/docs/src/pages/api/me/state.ts apps/docs/tests/integration/me-state.test.ts
git commit -m "feat(account): GET /api/me/state — single hydration call for authed pages"
```

---

## Task 7.2: Notes endpoint + drawer

**Files:**
- Create: `apps/docs/src/pages/api/notes/[lesson].ts` (PUT)
- Create: `apps/docs/src/components/lesson/NotesDrawer.svelte`
- Test: `apps/docs/tests/integration/notes.test.ts`

**Step 1: Endpoint.** PUT `{ body: string (max 50KB) }` → upsert one row per (user_id, lesson_slug). GET (added to `/api/me/state` body fetch path or direct route) returns `{ body, updatedAt }`.

**Step 2: NotesDrawer.** Toggle button "Notes" in lesson header → slides in right-side drawer with single `<textarea>`. Autosaves on idle (1.5s debounce). Shows last-saved timestamp.

**Step 3: Tests.** PUT upserts, oversize rejected, foreign access 403.

**Step 4: Commit.**

```bash
git add apps/docs/src/pages/api/notes apps/docs/src/components/lesson/NotesDrawer.svelte apps/docs/tests/integration/notes.test.ts
git commit -m "feat(notes): PUT endpoint + autosaving drawer per lesson"
```

---

# Phase 8 — Admin UI

## Task 8.1: Admin guard + dashboard

**Files:**
- Create: `apps/docs/src/server/admin.ts` — `requireAdmin()` returns 404 (not 403) for non-admins
- Create: `apps/docs/src/pages/admin/index.astro` — dashboard
- Test: `apps/docs/tests/integration/admin.test.ts`

**Step 1: requireAdmin.** Reads session, queries user role, returns 404 if not admin.

**Step 2: Dashboard.** Stats: total users, opted-in count, recent 10 drops. Links to /admin/drops/new and /admin/users.

**Step 3: Tests.** Non-admin → 404. Anon → 404. Admin → 200.

**Step 4: Commit.**

```bash
git add apps/docs/src/server/admin.ts apps/docs/src/pages/admin/index.astro apps/docs/tests/integration/admin.test.ts
git commit -m "feat(admin): guard + dashboard with stats"
```

---

## Task 8.2: Drop composer + send

**Files:**
- Create: `apps/docs/src/pages/admin/drops/new.astro`
- Create: `apps/docs/src/pages/admin/drops/[id].astro`
- Create: `apps/docs/src/pages/api/admin/drops.ts`
- Create: `apps/docs/src/components/admin/DropComposer.svelte`
- Test: `apps/docs/tests/integration/admin-drops.test.ts`

**Step 1: Composer.** Fields: lesson dropdown (auto-populate subject + teaser from MDX frontmatter), markdown body editor, "Send test to me" button, "Send to N opted-in" button (with confirm modal).

**Step 2: Endpoint.** `POST /api/admin/drops` `{ subject, body_md, lesson_slug?, test_only?: boolean }`:
- Insert `email_drop` row with `target_count`
- Iterate over opted-in users (or just self if `test_only`), call `sendDropEmail` per recipient
- Update `sent_count` as each succeeds (best-effort: log failures, don't fail the whole drop)
- Each email gets a unique unsubscribe link signed with HMAC

**Step 3: Detail page.** Lists drop + audience size + sent count + when + body preview.

**Step 4: Tests.** Test send only emails self; full send hits all opted-in; unsubscribe link for user X actually toggles `marketingOptIn` for X.

**Step 5: Commit.**

```bash
git add apps/docs/src/pages/admin/drops apps/docs/src/pages/api/admin apps/docs/src/components/admin/DropComposer.svelte apps/docs/tests/integration/admin-drops.test.ts
git commit -m "feat(admin): drop composer + send + detail page"
```

---

## Task 8.3: Admin user list + unsubscribe endpoint

**Files:**
- Create: `apps/docs/src/pages/admin/users.astro` (read-only)
- Create: `apps/docs/src/pages/api/unsubscribe.ts`
- Test: extend existing tests

**Step 1: User list.** Paginated table: email, role, marketing_opt_in, created_at, last lesson_view.lastSeenAt. No actions in v1.

**Step 2: Unsubscribe endpoint.** `GET /api/unsubscribe?token=…` — verify HMAC token via `verifyUnsubscribeToken`, flip `marketingOptIn = false`, render a friendly "you're unsubscribed" page. Idempotent.

**Step 3: Commit.**

```bash
git add apps/docs/src/pages/admin/users.astro apps/docs/src/pages/api/unsubscribe.ts
git commit -m "feat(admin): user list + public unsubscribe endpoint"
```

---

# Phase 9 — Capacity messaging + replace waitlist

## Task 9.1: Replace #waitlist on index.astro

**Files:**
- Modify: `apps/docs/src/pages/index.astro`

**Step 1: Edit.** Replace lines 184–208 (the `#waitlist` band) with the sign-up band copy from the design doc. Replace the hero CTA on line 43 ("Join the waitlist") with "Sign up — it's free." Keep "No signup to explore."

**Step 2: Remove the stale stub script** at the bottom of `index.astro` that was POSTing nowhere.

**Step 3: Visual smoke.** `pnpm dev`, scroll through `/`. Confirm the band looks right against `DESIGN.md`.

**Step 4: Commit.**

```bash
git add apps/docs/src/pages/index.astro
git commit -m "feat(home): replace waitlist with real signup band + alpha framing"
```

---

# Phase 10 — Visual + E2E test pass

## Task 10.1: Add Playwright config for docs E2E

**Files:**
- Create: `apps/docs/playwright.config.ts`
- Create: `apps/docs/tests/e2e/auth-and-progress.spec.ts`
- Modify: root `package.json` to wire up E2E

**Step 1: Playwright config.** Extends existing svelte-mafs setup. `baseURL` is `http://127.0.0.1:8788` (wrangler dev default). `webServer` runs `pnpm wrangler dev --local --persist-to=.wrangler-state` against the worker.

**Step 2: Test scenarios.**
1. Magic link signup: submit email → read latest token from D1 verification table directly → visit verify URL → see signed-in state.
2. Visit 3 lessons → return to `/courses/ml-math` → see "Continue where you left off" + 3 ✓.
3. Anon visits 2 lessons → sign up → those 2 lessons show as visited.
4. Bookmark a lesson → see it at `/me/bookmarks` → remove → it's gone.
5. Type in notes drawer → wait 2s → reload page → notes restored.
6. Sign out → progress UI gone → sign back in → progress UI back.

**Step 3: CI.** Add an E2E job in `.github/workflows/` (extend the existing one) that runs the docs E2E.

**Step 4: Commit.**

```bash
git add apps/docs/playwright.config.ts apps/docs/tests/e2e .github/workflows
git commit -m "test(e2e): playwright suite for auth + progress + bookmarks + notes"
```

---

## Task 10.2: Visual regression baselines

**Files:**
- Add Playwright `toHaveScreenshot()` to e2e specs
- Generate baselines

**Step 1: Add visual checks** to: signin, signup, /welcome, /me, course landing (signed-in vs signed-out), signed-in lesson, notes drawer open.

**Step 2: Generate baselines.** Run with `--update-snapshots` once, commit baselines.

**Step 3: Commit.**

```bash
git add apps/docs/tests/e2e/__screenshots__
git commit -m "test(visual): baseline screenshots for auth + dashboard + lesson"
```

---

# Phase 11 — Final verification + deploy + merge

## Task 11.1: Full local verification

**Step 1:** `pnpm -r typecheck` — expect 0 errors.
**Step 2:** `pnpm -r test` — expect all unit + integration green.
**Step 3:** `pnpm -F docs build` — expect clean build.
**Step 4:** `pnpm -F docs test:e2e` — expect all E2E green (using local wrangler dev).

If any fails: stop, fix, re-run before proceeding.

---

## Task 11.2: Apply remote D1 migration

**Step 1:** `cd apps/docs && pnpm dlx wrangler@latest d1 migrations apply tinker --remote`
Expected: All migrations applied.

**Step 2: Bootstrap admin role.** After your first prod signup:

```bash
pnpm dlx wrangler@latest d1 execute tinker --remote --command "update user set role='admin' where email='joshhunterduvar@gmail.com'"
```

(You won't have signed up yet — defer Step 2 until after deploy + first signup.)

---

## Task 11.3: Set production secrets

**Step 1:** From `apps/docs/`:

```bash
pnpm dlx wrangler@latest secret put BETTER_AUTH_SECRET
pnpm dlx wrangler@latest secret put RESEND_API_KEY
pnpm dlx wrangler@latest secret put GOOGLE_CLIENT_ID
pnpm dlx wrangler@latest secret put GOOGLE_CLIENT_SECRET
pnpm dlx wrangler@latest secret put GITHUB_CLIENT_ID
pnpm dlx wrangler@latest secret put GITHUB_CLIENT_SECRET
pnpm dlx wrangler@latest secret put UNSUBSCRIBE_HMAC_SECRET
pnpm dlx wrangler@latest secret put TURNSTILE_SECRET_KEY
```

For each, paste the value when prompted.

**Step 2: Add public env vars.** In `wrangler.jsonc`:

```jsonc
"vars": {
  "PUBLIC_SITE_URL": "https://learntinker.com",
  "TURNSTILE_SITE_KEY": "<your public site key>"
}
```

---

## Task 11.4: Deploy

**Step 1:** `pnpm -F docs build`
**Step 2:** `cd apps/docs && pnpm dlx wrangler@latest deploy`
**Step 3: Smoke test prod.** Visit `https://learntinker.com/signup`, sign up with your real email, verify the email actually arrives, click the link, complete welcome flow, view a lesson, see progress save, sign out, sign back in, see progress restored.
**Step 4: Bootstrap admin.** Run the SQL from Task 11.2 Step 2 now that your user exists.
**Step 5: Send a test drop** to yourself via `/admin/drops/new` to confirm the email flow.

---

## Task 11.5: Merge to main

**Step 1: Push the branch.**

```bash
git push -u origin feature/auth-v1
```

**Step 2: Open PR or merge directly.** User said "merge it as soon as you're done" — assuming direct merge to `main`:

```bash
git checkout main
git merge --no-ff feature/auth-v1 -m "feat: signup + auth v1"
git push origin main
```

**Step 3: Clean up the worktree** (after merge confirmed):

```bash
git worktree remove .worktrees/feature-auth-v1
git branch -d feature/auth-v1
```

(Skip the worktree removal until after the user has confirmed merge looks right.)

---

# Done definition

- `/signup`, `/signin`, `/verify`, `/welcome`, `/me`, `/me/bookmarks`, `/admin`, `/admin/drops/new` all work in prod.
- Magic link, Google OAuth, GitHub OAuth all complete a full sign-in.
- Progress saves on lesson view, restores on return, "Continue where you left off" appears.
- Bookmarks + notes + exercise persistence all round-trip.
- Drop emails send to opted-in users only; unsubscribe link works.
- Anonymous progress merges into account on first sign-in.
- Account deletion cascades and anonymizes email.
- Non-admins get 404 on every `/admin/*` route.
- All unit + integration + E2E tests pass in CI.
- Waitlist stub is fully removed from `index.astro`.
- Capacity copy ("alpha · 1 course live · new modules weekly · {n} learners") visible on home + course landing.

---

# Out of scope (deliberately not in this plan)

- Stripe / paid tier — Better Auth's Stripe plugin slots in later
- Account merge UI for users with two emails on different providers
- Markdown rendering in notes
- Multi-note per lesson
- Push notifications for course drops
- Admin user moderation (ban, role-change)
- Cross-device session listing / remote sign-out
- Soft-delete with 30-day reactivation window
- Load testing
- Cross-browser tests beyond Chromium
