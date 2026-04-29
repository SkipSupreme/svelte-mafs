import { drizzle } from 'drizzle-orm/d1';
import type { BaseSQLiteDatabase } from 'drizzle-orm/sqlite-core';
import * as schema from './schema';

/**
 * Structural DB type compatible with both D1 (production) and better-sqlite3
 * (tests). Using BaseSQLiteDatabase lets us pass either driver to functions
 * that just need a Drizzle SQLite client over our schema.
 */
export type DB = BaseSQLiteDatabase<'async' | 'sync', unknown, typeof schema>;

export function getDb(binding: D1Database): DB {
  if (!binding) {
    throw new Error('D1 binding missing — check wrangler.jsonc');
  }
  return drizzle(binding, { schema }) as unknown as DB;
}
