# Signup & auth — v1 design

Replace the stub waitlist on `index.astro` with a real signup flow. Build the user model, auth, progress tracking, bookmarks, notes, admin UI, and email infrastructure that gets Tinker out of "wait list" mode and into "real users have real accounts."

## Goal

- Anyone can sign up and have a persistent account.
- Logged-in users get progress tracking, bookmarks, notes, and persisted exercise state.
- Owner (Josh) can compose and send "new module live" emails to opted-in users from an admin UI.
- Free for v1. User model is shaped so a paid tier can be added later without a rewrite.
- Honest capacity messaging: "alpha · 1 course live · new modules weekly."

## Decisions locked in

| Question | Choice |
|---|---|
| Free vs paid | Free now, paid later (build user model with future entitlements in mind) |
| Auth ownership | Self-hosted via Better Auth on Cloudflare D1 |
| Auth methods | Magic links + Google OAuth + GitHub OAuth |
| Scope of signed-in experience | Full: visited lessons, completion, exercise answers, bookmarks, notes |
| Drop emails | Full admin UI under `/admin`, not a CLI |
| Show learner count on course landing | Yes — easy to hide later |

## Architecture

### Stack additions

- `better-auth` — auth library (magicLink, google, github plugins)
- `drizzle-orm` + `drizzle-kit` — D1 ORM and migrations
- `resend` — transactional + marketing email
- `zod` — input validation on every endpoint
- Cloudflare Turnstile — invisible bot check on signup form

Roughly +600KB to the dev tree, ~100KB shipped to the Worker after tree-shaking.

### Output mode

`apps/docs/astro.config.mjs` flips `output: 'static'` → `output: 'server'`. Every existing page that doesn't need request context gets `export const prerender = true` at the top so it stays edge-cached. Only API routes and authed pages run on the Worker per request.

### Database

Cloudflare D1, bound as `DB` in `wrangler.jsonc`. Local dev uses Wrangler's local D1 emulator. Tests use a fresh in-memory D1 per run. Schema in `apps/docs/src/server/schema.ts` (Drizzle). Migrations in `apps/docs/migrations/`, applied via `wrangler d1 migrations apply`.

### File map

```
apps/docs/
  src/
    server/
      auth.ts          better-auth config + provider plugins
      db.ts            drizzle client factory bound to env.DB
      schema.ts        drizzle schema for all tables
      email.ts         resend wrapper, template renderers, audience filters
      ratelimit.ts     sliding-window limiter keyed in D1
      admin.ts         requireAdmin() middleware
    pages/
      api/
        auth/[...all].ts            better-auth handler (all methods)
        progress.ts                 POST view, complete
        progress/merge.ts           POST anon → user merge
        exercises/answer.ts         POST exercise attempt
        bookmarks/index.ts          POST create
        bookmarks/[id].ts           DELETE
        notes/[lesson].ts           PUT upsert note
        me/index.ts                 DELETE account
        me/state.ts                 GET single hydration call
        unsubscribe.ts              GET HMAC-signed unsubscribe
        admin/drops.ts              POST send a drop
      signup.astro
      signin.astro
      verify.astro
      welcome.astro
      me.astro
      me/bookmarks.astro
      admin/index.astro             dashboard
      admin/drops/new.astro         composer
      admin/drops/[id].astro        detail
      admin/users.astro             user list
    components/
      auth/
        AuthForm.svelte             email field + 3 provider buttons
        UserMenu.svelte             avatar dropdown in nav
        AuthGate.svelte             "sign in to use this" wrapper
      lesson/
        ProgressBeacon.svelte       fires view + visibility events
        BookmarkButton.svelte
        NotesDrawer.svelte
        CompleteToggle.svelte
      admin/
        DropComposer.svelte         compose, preview, test send, send
  migrations/
    0001_init.sql
  wrangler.jsonc                    D1 binding + secrets list
```

## Data model

### Auth tables (Better Auth managed)

```
user
  id              text pk
  email           text unique not null
  email_verified  boolean default false
  name            text
  image           text                              -- avatar from OAuth
  role            text default 'user'               -- 'user' | 'admin'
  created_at      datetime not null
  updated_at      datetime not null

session
  id              text pk
  user_id         text fk -> user(id) on delete cascade
  token           text unique not null
  expires_at      datetime not null
  ip              text
  user_agent      text
  created_at      datetime not null

account                                              -- one row per linked provider
  id              text pk
  user_id         text fk -> user(id) on delete cascade
  provider_id     text not null                      -- 'google' | 'github' | 'magic-link'
  account_id      text not null                      -- provider's user id
  unique(provider_id, account_id)

verification                                         -- magic link tokens
  id              text pk
  identifier      text not null                      -- email
  value           text not null                      -- token hash
  expires_at      datetime not null
  created_at      datetime not null
```

### Product tables

```
user_profile
  user_id          text pk fk -> user(id) on delete cascade
  display_name     text
  marketing_opt_in boolean default false
  onboarded_at     datetime

lesson_view
  user_id         text fk -> user(id) on delete cascade
  course_slug     text not null                      -- 'ml-math'
  module_slug     text not null                      -- 'm5-calculus'
  lesson_slug     text not null                      -- 'what-is-a-derivative'
  first_seen_at   datetime not null
  last_seen_at    datetime not null
  view_count      integer default 1
  completed_at    datetime
  primary key (user_id, lesson_slug)
  index (user_id, course_slug, last_seen_at desc)

exercise_answer
  id              text pk
  user_id         text fk -> user(id) on delete cascade
  lesson_slug     text not null
  exercise_id     text not null                      -- stable id authored in MDX
  answer_json     text not null
  is_correct      boolean
  attempt_no      integer default 1
  created_at      datetime not null
  index (user_id, lesson_slug)

bookmark
  id              text pk
  user_id         text fk -> user(id) on delete cascade
  lesson_slug     text not null
  anchor          text
  created_at      datetime not null
  unique(user_id, lesson_slug, anchor)

note
  user_id         text fk -> user(id) on delete cascade
  lesson_slug     text not null
  body            text not null default ''
  updated_at      datetime not null
  primary key (user_id, lesson_slug)

email_drop
  id                text pk
  subject           text not null
  body_md           text not null
  course_slug       text
  module_slug       text
  lesson_slug       text
  target_count      integer not null                 -- audience size at send time
  sent_count        integer not null                 -- successful sends
  sent_at           datetime not null
  sent_by_user_id   text fk -> user(id)
```

### Shape rationale

- `lesson_view` uses composite PK `(user_id, lesson_slug)` so the progress beacon is a single idempotent `INSERT … ON CONFLICT DO UPDATE`. No race conditions, no read-then-write.
- Course/module slugs denormalized into `lesson_view` so course landing renders with no joins.
- `exercise_answer` keeps every attempt rather than overwriting — supports "you've tried this 3 times" later. Latest row is current state.
- `note` is one body per lesson, not multi-note per lesson. Single textarea drawer. Splits later if needed.
- `marketing_opt_in` lives in `user_profile`, not on `user`, so "delete my marketing prefs" doesn't touch auth.
- Lesson slugs are stable in MDX frontmatter. Renames ship as one-shot SQL migrations.

## Auth flows

### Magic link

1. User submits email at `/signup` or `/signin` → `POST /api/auth/sign-in/magic-link`.
2. Edge rate-limit check (per email: 3/hour; per IP: 10/hour, sliding window in D1).
3. Better Auth generates token, stores it in `verification` (15-min TTL, single-use), Resend sends the link.
4. User clicks `https://learntinker.com/verify?token=…` → Better Auth verifies, marks `email_verified = true`, creates session, sets cookie, 302 to `/` or `?next=` target.
5. First-time only: create `user_profile` row, redirect to `/welcome` with the marketing opt-in checkbox.

### OAuth (Google, GitHub)

1. Click "Continue with X" → 302 to provider → callback at `/api/auth/callback/[provider]`.
2. Better Auth either creates user (new) or links to existing user (matching verified email).
3. Same first-time `/welcome` redirect.

### Edge cases

| Case | Handling |
|---|---|
| Same email signs up via magic link, later via Google | Better Auth links the Google `account` row to the existing `user`. Email already verified → no friction. |
| Different email per provider for same person | Treated as separate users. `/me` will offer manual account merge as a follow-up. |
| Magic link opened on a different device | Works fine — token is the auth, not the device. Email body shows requesting device + IP. |
| Magic link clicked twice | First click consumes the token; second click → expired-token page with "Send another link". |
| Magic link expired (>15 min) | Same expired-token page → resend flow. No silent retry. |
| Rate-limit hit | Friendly 429 page: "Too many sign-in attempts. Wait 10 minutes or use Google/GitHub." Counter keyed in D1 with TTL row. |
| Session cookie present but session row missing/expired | Treated as logged out. Cookie cleared on next response. |
| Anonymous progress (localStorage) at sign-up | First authed request POSTs the localStorage payload to `/api/progress/merge`. Server idempotently upserts; refuses to overwrite a `completed_at` already set. |
| Account deletion at `/me` | Cascade-deletes everything. `user.email` set to `deleted-{uuid}@deleted.local` so the address can be reused. Confirmation email via Resend. |
| Cloudflare Worker cold start + D1 RTT | Auth handler in single region close to user; D1 reads ~5ms. Sessions cached in Worker memory for the request. |
| CSRF | Better Auth double-submit cookie. State-changing endpoints require session cookie + `X-Tinker-CSRF` header matching cookie value. |
| Bot/spam signups | Cloudflare Turnstile (invisible) on the email entry. |

## Progress, exercises, bookmarks, notes

### Progress beacon

`<ProgressBeacon lessonSlug courseSlug moduleSlug />` mounts on every lesson page. On mount → `POST /api/progress/view` (debounced 2s). On `visibilitychange → hidden` → another POST to update `last_seen_at`. Anonymous users skip the network call but write the same payload to `localStorage` keyed `tinker:progress:v1`. On first authed page load, that payload is replayed by the merge endpoint and cleared.

### Completion

Two paths:
1. A lesson's terminal widget calls `POST /api/progress/complete` with the lesson slug.
2. The `<CompleteToggle>` in the lesson footer lets the user manually mark complete.

`completed_at` only moves forward. Re-marking incomplete is allowed via the UI but never via the beacon.

### "Continue where you left off"

Course landing runs a single query:

```sql
select * from lesson_view
where user_id = ? and course_slug = 'ml-math'
order by last_seen_at desc
limit 1
```

Result becomes the primary CTA at the top: "Continue: *What is a derivative?*". Below that, the existing arc/module tree gets per-module checkmarks driven by:

```sql
select lesson_slug, completed_at from lesson_view where user_id = ?
```

### Exercises

Widget contract gains two methods: `serializeAnswer()` and `restoreAnswer(json)`. The lesson page wraps each exercise in a thin Svelte client that:
- On submit: `POST /api/exercises/answer` with `{ lesson_slug, exercise_id, answer_json, is_correct }`.
- On mount: calls `restoreAnswer` with the latest persisted attempt.

Widgets that don't opt in are unaffected. `exercise_id` is authored in MDX and must be stable per lesson.

### Bookmarks

Bookmark icon in the lesson header. Click → `POST /api/bookmarks` with `{ lesson_slug, anchor: window.location.hash || null }`. Bookmark list at `/me/bookmarks`. Remove via `DELETE /api/bookmarks/:id`.

### Notes

Right-side drawer toggled by a "Notes" button. Single textarea per lesson, autosaves on idle (1.5s debounce) via `PUT /api/notes/:lesson_slug`. Plaintext for v1; markdown rendering is a follow-up.

### Endpoint contract

```
POST   /api/progress/view             { lesson_slug, course_slug, module_slug }
POST   /api/progress/complete         { lesson_slug }
POST   /api/progress/merge            { entries: LessonView[] }
POST   /api/exercises/answer          { lesson_slug, exercise_id, answer_json, is_correct }
GET    /api/me/state?course=ml-math   → { progress, exercises, bookmarks, notes_index }
POST   /api/bookmarks                 { lesson_slug, anchor? }
DELETE /api/bookmarks/:id
PUT    /api/notes/:lesson_slug        { body }
DELETE /api/me                        // hard delete
GET    /api/unsubscribe?token=…       // HMAC-signed, no auth
POST   /api/admin/drops               { subject, body_md, lesson_slug?, test_only? }
```

All endpoints: session-required (except `/api/auth/*` and unsubscribe), CSRF-protected, Zod-validated, rate-limited at 60 req/min per user. Consistent error envelope: `{ error: { code, message } }`. `/api/me/state` is the single hydration call — every authed page fetches it once and feeds client components, avoiding N requests on lesson load.

## Capacity messaging + email

### Index page

Replace the existing waitlist band (lines 184–208 of `apps/docs/src/pages/index.astro`) with a sign-up band:

> **Tinker is open. Come build math.**
> One course live today: *Math for Machine Learning*. New modules every week. Sign up to track your progress, save notes, and get the heads-up when new courses ship.
>
> [ Continue with Google ] [ Continue with GitHub ]
> — or your email — [_______________] [ Send link ]

Hero CTA (line 43): "Join the waitlist" → "Sign up — it's free." The "No signup to explore" line stays.

### Course landing

Thin alpha banner at the top:

> 🌱 *Alpha · 1 course live · new modules weekly · {n} learners*

`{n}` is `select count(*) from user` rendered server-side per request.

### Email infrastructure

| Email | Trigger | Audience |
|---|---|---|
| Magic link | sign-in request | self |
| Welcome | first-time `email_verified = true` | self |
| New module live | admin send via `/admin/drops/new` | `marketing_opt_in = true` |
| Account deleted | `DELETE /api/me` | self |

All sends go through `apps/docs/src/server/email.ts`:
- `sendTransactional(to, template, props)` — magic link, welcome, deletion confirmation.
- `sendMarketing(audience, template, props)` — drops only. Filters to opted-in users at query time.

Templates are React-Email components rendered to HTML at build time, stored as static strings the Worker ships without bundling React at runtime.

### Compliance

Every marketing email has a one-click unsubscribe link → `GET /api/unsubscribe?token=…` that flips `marketing_opt_in = false`. Token is HMAC-signed with `UNSUBSCRIBE_HMAC_SECRET`, no auth needed. Transactional emails (magic link, welcome, deletion) are exempt from unsubscribe but only ever sent on user-initiated action.

## Admin UI

Surface gated by `requireAdmin()` middleware. Non-admins get **404** (not 403) so the surface stays invisible. Bootstrap your own admin role via a one-line SQL migration after first signup: `update user set role = 'admin' where email = 'joshhunterduvar@gmail.com'`.

| Route | Purpose |
|---|---|
| `/admin` | Dashboard: signup count, opt-in count, last 10 drops, quick links |
| `/admin/drops/new` | Composer: pick a lesson, auto-populate subject + teaser from MDX frontmatter, preview rendered email, "send test to me", "send to {n} opted-in users" with confirm modal |
| `/admin/drops/:id` | Detail view of a past send (audience size, sent count, when, which lesson) |
| `/admin/users` | Paginated user list, last seen, opt-in status. Read-only in v1. |

Drop send writes a row to `email_drop` before sending, then updates `sent_count` as Resend confirms each send. Failed sends per-recipient are logged but don't fail the whole drop.

## Testing strategy

Three layers, real coverage on the parts that matter.

### Unit (Vitest)

Pure-logic boundaries:
- `src/server/auth.ts` — provider config, callback URL builder, session-cookie options
- `src/server/ratelimit.ts` — sliding-window math, key derivation, reset behavior
- `src/server/email.ts` — template rendering, audience filter, HMAC signing/verification
- Zod input schemas for every endpoint
- Anonymous-progress merge function (idempotency, conflict resolution with `completed_at`)

Run with `pnpm -F docs test:unit`. Target: 90%+ on `src/server/**`.

### Integration (Vitest + Miniflare)

Spin up the Worker locally with a fresh in-memory D1 per test. Migrations applied at suite start.

- Magic link: request → token row → click verify → session cookie set → `/api/me/state` returns user
- OAuth callback: stub provider, hit callback, verify `user` + `account` + `session` rows
- Every edge case from the auth section: expired token, replayed token, rate-limit boundary, email already linked to other provider, anonymous-progress merge
- Every endpoint: unauthenticated → 401, valid session → 200, foreign user_id in payload → 403
- CSRF: missing header → 403, wrong header → 403, correct → 200
- Account deletion: cascades correctly, email anonymized, session invalidated
- Admin gating: non-admin hits `/admin/*` → 404 (not 403)

### E2E (Playwright)

Real browser, real Worker (`wrangler dev`), real local D1, mocked Resend. Reuses existing Playwright setup in `packages/svelte-mafs/`.

- Signup via magic link (test reads token from D1's `verification` table directly, since email is mocked)
- Signup via Google + GitHub (provider sandbox/stub)
- "Continue where you left off" appears after viewing 3 lessons
- Anonymous → sign up → previously-viewed lessons show as visited
- Bookmark create → appears at `/me/bookmarks` → delete
- Notes autosave → reload → notes restored
- Sign out clears progress UI; re-sign-in restores it
- Visual regression on signup, signin, /me, course landing with progress

### CI

Add a `db-migrate` step before the existing E2E job. Tests run against per-job D1 instance; no shared state.

### Out of scope for v1 testing

- Load testing (premature; Cloudflare D1 + Workers handle our scale)
- Cross-browser beyond Chromium (existing Playwright is Chromium-only)
- Email deliverability (Resend's responsibility)

## Rollout sequence

Each step is independently shippable; the site stays working at every commit.

1. **Foundation.** D1 binding in `wrangler.jsonc`. `output: 'static'` → `output: 'server'` with `prerender = true` on every existing page (zero behavior change). Install Drizzle + Better Auth + Resend + Zod. Write schema, generate migration, apply locally.
2. **Auth backbone.** Wire Better Auth at `/api/auth/[...all]`. Register Google + GitHub OAuth apps (callbacks `https://learntinker.com/api/auth/callback/{provider}` + a localhost variant). Resend account, magic-link template, secrets stored via `wrangler secret put`.
3. **Auth UI.** Build `/signup`, `/signin`, `/verify`, `/welcome`. Add `<UserMenu>` to nav. Site looks identical to logged-out users at this point.
4. **Progress.** Ship `<ProgressBeacon>` and `/api/progress/*`. Add "Continue where you left off" CTA. Per-module checkmarks. Anonymous → authenticated merge endpoint.
5. **Exercises, bookmarks, notes.** Each is an independent slice — endpoint + component + `/me` listing. Any order.
6. **Admin UI.** `/admin` dashboard, drop composer, drop detail, user list. Bootstrap own admin role via SQL.
7. **Replace the waitlist.** Rewrite `#waitlist` band on `index.astro` into the sign-up band. Change hero CTA. Add alpha banner to course landing. Only user-visible step. ~30-line diff.
8. **Testing pass.** Fill in unit + integration + E2E. CI gate.
9. **Deploy.** `pnpm -F docs build && pnpm dlx wrangler@latest deploy` from `apps/docs/`. Smoke-test signup against prod.

Steps 1–6 are invisible to users. Step 7 is the one-way door, but reverting is `git revert`.

## Secrets needed

Set via `wrangler secret put`. Localdev uses `.dev.vars` (gitignored).

- `BETTER_AUTH_SECRET` — generate with `openssl rand -hex 32`
- `RESEND_API_KEY`
- `GOOGLE_CLIENT_ID` + `GOOGLE_CLIENT_SECRET`
- `GITHUB_CLIENT_ID` + `GITHUB_CLIENT_SECRET`
- `UNSUBSCRIBE_HMAC_SECRET` — generate with `openssl rand -hex 32`
- `TURNSTILE_SECRET_KEY` (and matching public site key in client)

## Out-of-band setup needed from owner

1. **Google OAuth app** — Google Cloud Console → APIs & Services → Credentials → Create OAuth client ID (Web). Authorized redirect URIs: `https://learntinker.com/api/auth/callback/google`, `http://localhost:4321/api/auth/callback/google`.
2. **GitHub OAuth app** — GitHub → Settings → Developer settings → OAuth Apps → New. Same callback URLs.
3. **Resend account** — sign up, verify `learntinker.com` (DKIM/SPF/DMARC records), generate API key.
4. **Cloudflare Turnstile site** — Cloudflare dashboard → Turnstile → Add site. Use invisible widget mode.

## Out of scope for v1

- Paid tier, Stripe integration (Better Auth's Stripe plugin will slot in cleanly when needed)
- Account merge UI for users who signed up with two emails
- Markdown rendering for notes
- Multi-note-per-lesson
- Mobile push notifications for course drops
- Admin user management beyond read-only listing
- Cross-device session listing / remote sign-out
- Soft-delete and 30-day reactivation window for deleted accounts
