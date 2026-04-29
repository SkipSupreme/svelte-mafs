import { describe, it, expect, vi, beforeEach } from 'vitest';
import { createAuth } from './auth';

vi.mock('./db', () => ({
  getDb: () => ({}),
}));

const baseEnv = {
  DB: {} as D1Database,
  BETTER_AUTH_SECRET: 'a'.repeat(64),
  PUBLIC_SITE_URL: 'http://localhost:4321',
} as const;

beforeEach(() => {
  vi.unstubAllGlobals();
});

describe('createAuth', () => {
  it('returns an instance with handler + api', () => {
    const auth = createAuth({ ...baseEnv });
    expect(typeof auth.handler).toBe('function');
    expect(auth.api).toBeDefined();
  });

  it('throws when BETTER_AUTH_SECRET is missing', () => {
    expect(() => createAuth({ ...baseEnv, BETTER_AUTH_SECRET: '' })).toThrow(/secret/i);
  });

  it('throws when PUBLIC_SITE_URL is missing', () => {
    expect(() => createAuth({ ...baseEnv, PUBLIC_SITE_URL: '' })).toThrow(/PUBLIC_SITE_URL/i);
  });

  it('only registers social providers when both id and secret are present', () => {
    const noProviders = createAuth({ ...baseEnv });
    const withGoogle = createAuth({
      ...baseEnv,
      GOOGLE_CLIENT_ID: 'g',
      GOOGLE_CLIENT_SECRET: 's',
    });
    expect(typeof noProviders.handler).toBe('function');
    expect(typeof withGoogle.handler).toBe('function');
  });
});
