import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  sendDropEmail,
  signUnsubscribeToken,
  verifyUnsubscribeToken,
} from './email';

const fetchMock = vi.fn();

beforeEach(() => {
  fetchMock.mockReset();
  fetchMock.mockResolvedValue(new Response(JSON.stringify({ id: 'msg_123' }), { status: 200 }));
  globalThis.fetch = fetchMock as unknown as typeof fetch;
});

describe('sendDropEmail', () => {
  it('includes List-Unsubscribe headers', async () => {
    await sendDropEmail('re_test', 'a@b.co', 'New module', '<p>hi</p>', 'https://x/unsub?token=t');
    const [, init] = fetchMock.mock.calls[0];
    const body = JSON.parse(init.body as string);
    expect(body.headers['List-Unsubscribe']).toBe('<https://x/unsub?token=t>');
    expect(body.headers['List-Unsubscribe-Post']).toBe('List-Unsubscribe=One-Click');
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

  it('rejects malformed tokens', async () => {
    expect(await verifyUnsubscribeToken('garbage', 'secret')).toBeNull();
    expect(await verifyUnsubscribeToken('', 'secret')).toBeNull();
    expect(await verifyUnsubscribeToken('.justsig', 'secret')).toBeNull();
  });

  it('rejects a tampered userId', async () => {
    const tok = await signUnsubscribeToken('user-123', 'secret');
    const [, sig] = tok.split('.');
    expect(await verifyUnsubscribeToken(`user-456.${sig}`, 'secret')).toBeNull();
  });
});
