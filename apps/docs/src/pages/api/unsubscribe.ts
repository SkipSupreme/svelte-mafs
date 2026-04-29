import type { APIRoute } from 'astro';
import { eq } from 'drizzle-orm';
import { getDb } from '../../server/db';
import { userProfile } from '../../server/schema';
import { verifyUnsubscribeToken } from '../../server/email';
import { getEnv } from '../../server/env';

export const prerender = false;

const HTML_HEADERS = { 'content-type': 'text/html; charset=utf-8' };

const STYLES = `
  html { font-family: ui-sans-serif, system-ui, sans-serif; background: #fdfdfc; color: #17181a; }
  main { max-width: 480px; margin: 4rem auto; padding: 2rem; text-align: center; }
  h1 { font-size: 1.5rem; margin-bottom: 0.5rem; }
  p { color: #585a60; }
  button { font: inherit; padding: 0.7rem 1.25rem; border-radius: 999px; border: 1.5px solid #17181a; background: #17181a; color: #fdfdfc; cursor: pointer; }
  button:hover { transform: translateY(-1px); }
  a { color: #17181a; }
`;

function page(body: string, status = 200): Response {
  return new Response(
    `<!doctype html><html lang="en"><head><meta charset="utf-8" /><title>Unsubscribe</title><style>${STYLES}</style></head><body><main>${body}</main></body></html>`,
    { status, headers: HTML_HEADERS },
  );
}

// Must be a function, not a const Response — Response bodies are streams and
// re-serving the same Response object across requests throws after the first.
const bad = () =>
  page(
    `<h1>This link is no longer valid.</h1><p>If you'd still like to unsubscribe, <a href="/me">visit your account page</a>.</p>`,
    400,
  );

// GET shows a confirmation form. We deliberately do NOT mutate state on GET so
// link prefetchers (Outlook safe-link, Gmail prefetch, anti-virus URL scanners,
// browser link previews) can't unsubscribe people. RFC 8058 specifies POST for
// the one-click endpoint via the `List-Unsubscribe-Post` header.
export const GET: APIRoute = async ({ url }) => {
  const env = getEnv();
  const token = url.searchParams.get('token');
  if (!token || !env.UNSUBSCRIBE_HMAC_SECRET) return bad();
  const userId = await verifyUnsubscribeToken(token, env.UNSUBSCRIBE_HMAC_SECRET);
  if (!userId) return bad();
  // Don't echo userId back to the client; just confirm the token is valid.
  return page(
    `<h1>Unsubscribe from module-launch emails?</h1>
     <p>Sign-in and account emails will still go through.</p>
     <form method="POST" action="/api/unsubscribe">
       <input type="hidden" name="token" value="${escapeAttr(token)}" />
       <button type="submit">Confirm unsubscribe</button>
     </form>
     <p style="margin-top:2rem;font-size:0.85rem"><a href="/me">Or manage email preferences in your account</a></p>`,
  );
};

export const POST: APIRoute = async ({ request, url }) => {
  const env = getEnv();
  if (!env.UNSUBSCRIBE_HMAC_SECRET) return bad();

  // RFC 8058 sends `List-Unsubscribe=One-Click` as form data. Browser form
  // submissions also use form data. JSON or query-string clients also work.
  let token = url.searchParams.get('token');
  if (!token) {
    const ct = request.headers.get('content-type') ?? '';
    if (ct.includes('application/x-www-form-urlencoded') || ct.includes('multipart/form-data')) {
      const form = await request.formData();
      token = (form.get('token') as string | null) ?? null;
    } else if (ct.includes('application/json')) {
      try {
        const body = (await request.json()) as { token?: string };
        token = body?.token ?? null;
      } catch {
        token = null;
      }
    }
  }
  if (!token) return bad();

  const userId = await verifyUnsubscribeToken(token, env.UNSUBSCRIBE_HMAC_SECRET);
  if (!userId) return bad();

  const db = getDb(env.DB);
  await db.update(userProfile).set({ marketingOptIn: false }).where(eq(userProfile.userId, userId));

  return page(
    `<h1>You're unsubscribed.</h1>
     <p>You won't receive any more module-launch emails from Tinker. Sign-in and account emails will still go through. <a href="/me">Visit your account</a> to opt back in any time.</p>`,
  );
};

function escapeAttr(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
