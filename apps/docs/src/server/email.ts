import {
  magicLinkTemplate,
  welcomeTemplate,
  dropTemplate,
  accountDeletedTemplate,
} from './email-templates';

const RESEND_URL = 'https://api.resend.com/emails';
const FROM = 'Tinker <hello@learntinker.com>';

interface ResendPayload {
  from: string;
  to: string[];
  subject: string;
  html: string;
  headers?: Record<string, string>;
}

async function postResend(apiKey: string, payload: ResendPayload): Promise<{ id: string }> {
  const res = await fetch(RESEND_URL, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Resend ${res.status}: ${text}`);
  }
  return res.json() as Promise<{ id: string }>;
}

export async function sendMagicLinkEmail(apiKey: string, to: string, url: string) {
  return postResend(apiKey, {
    from: FROM,
    to: [to],
    subject: 'Sign in to Tinker',
    html: magicLinkTemplate({ url }),
  });
}

export async function sendWelcomeEmail(apiKey: string, to: string) {
  return postResend(apiKey, {
    from: FROM,
    to: [to],
    subject: 'Welcome to Tinker',
    html: welcomeTemplate(),
  });
}

export async function sendDropEmail(
  apiKey: string,
  to: string,
  subject: string,
  bodyHtml: string,
  unsubUrl: string,
) {
  return postResend(apiKey, {
    from: FROM,
    to: [to],
    subject,
    html: dropTemplate({ bodyHtml, unsubUrl }),
    headers: {
      'List-Unsubscribe': `<${unsubUrl}>`,
      'List-Unsubscribe-Post': 'List-Unsubscribe=One-Click',
    },
  });
}

export async function sendAccountDeletedEmail(apiKey: string, to: string) {
  return postResend(apiKey, {
    from: FROM,
    to: [to],
    subject: 'Your Tinker account has been deleted',
    html: accountDeletedTemplate(),
  });
}

// HMAC-signed unsubscribe tokens. Lets us verify "this is the same user we
// signed for" without a DB lookup on every unsubscribe click.

const enc = new TextEncoder();

function bytesToB64Url(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf);
  let bin = '';
  for (let i = 0; i < bytes.byteLength; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function b64UrlToBytes(s: string): ArrayBuffer {
  const pad = s.length % 4 === 0 ? '' : '='.repeat(4 - (s.length % 4));
  const bin = atob(s.replace(/-/g, '+').replace(/_/g, '/') + pad);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out.buffer;
}

async function importHmacKey(secret: string): Promise<CryptoKey> {
  return crypto.subtle.importKey(
    'raw',
    enc.encode(secret),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign', 'verify'],
  );
}

export async function signUnsubscribeToken(userId: string, secret: string): Promise<string> {
  const key = await importHmacKey(secret);
  const sig = await crypto.subtle.sign('HMAC', key, enc.encode(userId));
  return `${userId}.${bytesToB64Url(sig)}`;
}

export async function verifyUnsubscribeToken(
  token: string,
  secret: string,
): Promise<string | null> {
  const dot = token.indexOf('.');
  if (dot < 1) return null;
  const userId = token.slice(0, dot);
  const sigB64 = token.slice(dot + 1);
  if (!userId || !sigB64) return null;
  try {
    const key = await importHmacKey(secret);
    const ok = await crypto.subtle.verify('HMAC', key, b64UrlToBytes(sigB64), enc.encode(userId));
    return ok ? userId : null;
  } catch {
    return null;
  }
}
