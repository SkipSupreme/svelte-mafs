import { betterAuth } from 'better-auth';
import { drizzleAdapter } from 'better-auth/adapters/drizzle';
import { magicLink } from 'better-auth/plugins';
import { getDb, type DB } from './db';
import { sendMagicLinkEmail } from './email';

export interface AuthEnv {
  DB: D1Database;
  BETTER_AUTH_SECRET: string;
  GOOGLE_CLIENT_ID?: string;
  GOOGLE_CLIENT_SECRET?: string;
  GITHUB_CLIENT_ID?: string;
  GITHUB_CLIENT_SECRET?: string;
  RESEND_API_KEY?: string;
  PUBLIC_SITE_URL: string;
}

export type Auth = ReturnType<typeof createAuth>;

/**
 * Create a Better Auth instance for the current request.
 * Configured here are: D1-backed Drizzle adapter, magic-link plugin,
 * Google + GitHub OAuth (if their secrets are present), 30-day sessions
 * with daily refresh, secure cookies in production.
 */
export function createAuth(env: AuthEnv) {
  if (!env.BETTER_AUTH_SECRET) {
    throw new Error('BETTER_AUTH_SECRET is required');
  }
  if (!env.PUBLIC_SITE_URL) {
    throw new Error('PUBLIC_SITE_URL is required');
  }

  const db = getDb(env.DB);
  const isHttps = env.PUBLIC_SITE_URL.startsWith('https://');

  const socialProviders: Record<string, { clientId: string; clientSecret: string }> = {};
  if (env.GOOGLE_CLIENT_ID && env.GOOGLE_CLIENT_SECRET) {
    socialProviders.google = {
      clientId: env.GOOGLE_CLIENT_ID,
      clientSecret: env.GOOGLE_CLIENT_SECRET,
    };
  }
  if (env.GITHUB_CLIENT_ID && env.GITHUB_CLIENT_SECRET) {
    socialProviders.github = {
      clientId: env.GITHUB_CLIENT_ID,
      clientSecret: env.GITHUB_CLIENT_SECRET,
    };
  }

  return betterAuth({
    database: drizzleAdapter(db, { provider: 'sqlite' }),
    secret: env.BETTER_AUTH_SECRET,
    baseURL: env.PUBLIC_SITE_URL,
    trustedOrigins: [env.PUBLIC_SITE_URL],
    emailAndPassword: { enabled: false },
    socialProviders,
    plugins: [
      magicLink({
        expiresIn: 60 * 15, // 15 minutes
        disableSignUp: false,
        sendMagicLink: async ({ email, url }) => {
          if (!env.RESEND_API_KEY) {
            // In local dev without Resend, log to console so the
            // developer can paste the URL directly. In tests we mock fetch
            // so this branch only runs when no key is set.
            console.log('[magic-link] (no RESEND_API_KEY set) ', email, url);
            return;
          }
          await sendMagicLinkEmail(env.RESEND_API_KEY, email, url);
        },
      }),
    ],
    advanced: {
      cookiePrefix: '__Secure-tinker',
      useSecureCookies: isHttps,
      defaultCookieAttributes: {
        sameSite: 'lax',
        secure: isHttps,
        httpOnly: true,
      },
    },
    session: {
      expiresIn: 60 * 60 * 24 * 30, // 30 days
      updateAge: 60 * 60 * 24, // refresh once per day
    },
    user: {
      additionalFields: {
        role: { type: 'string', defaultValue: 'user', input: false },
      },
    },
  });
}

export type SessionContext = {
  user: {
    id: string;
    email: string;
    name?: string | null;
    image?: string | null;
    role?: string;
    emailVerified?: boolean;
  };
  session: {
    id: string;
    userId: string;
    expiresAt: Date;
  };
};

export async function getSession(
  auth: Auth,
  headers: Headers,
): Promise<SessionContext | null> {
  const result = await auth.api.getSession({ headers });
  if (!result || !result.user || !result.session) return null;
  return result as unknown as SessionContext;
}
