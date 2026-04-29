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
