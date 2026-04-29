/**
 * Map ?error= URL params from Better Auth callbacks into user-facing copy.
 * Never echo the raw server message — it can leak Better Auth / Resend
 * internals.
 */
const TOKEN_ERRORS = new Set(['INVALID_TOKEN', 'EXPIRED_TOKEN']);

export function resolveAuthError(searchParams: URLSearchParams): string {
  const errorParam = searchParams.get('error') ?? '';
  if (!errorParam) return '';
  if (TOKEN_ERRORS.has(errorParam)) {
    return 'Your sign-in link has expired or already been used. Send a new one below.';
  }
  return 'Something went wrong. Try again.';
}
