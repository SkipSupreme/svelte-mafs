import Database from 'better-sqlite3';
import { drizzle } from 'drizzle-orm/better-sqlite3';
import * as schema from '../../src/server/schema';
import { readFileSync, readdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));

export interface TestDb {
  client: ReturnType<typeof drizzle<typeof schema>>;
  sqlite: Database.Database;
  close: () => void;
}

export function makeTestDb(): TestDb {
  const sqlite = new Database(':memory:');
  sqlite.pragma('foreign_keys = ON');
  const client = drizzle(sqlite, { schema });
  const dir = join(__dirname, '../../migrations');
  let files: string[] = [];
  try {
    files = readdirSync(dir).filter((f) => f.endsWith('.sql')).sort();
  } catch {
    files = [];
  }
  for (const f of files) {
    const sql = readFileSync(join(dir, f), 'utf8');
    for (const stmt of sql.split('--> statement-breakpoint').map((s) => s.trim()).filter(Boolean)) {
      sqlite.prepare(stmt).run();
    }
  }
  return { client, sqlite, close: () => sqlite.close() };
}
