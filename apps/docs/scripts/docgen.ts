#!/usr/bin/env tsx
/**
 * Docgen for svelte-mafs public API.
 *
 * Reads `packages/svelte-mafs/src/index.ts`, follows each `export … from "./x.js"`
 * back to its source (`./x.ts` or `./x.svelte`), extracts JSDoc + signatures,
 * and writes one MDX page per source module to `src/pages/api/<module>.mdx`
 * plus a manifest the index page renders.
 *
 * Design notes:
 * - Regex over TS AST: public-API re-exports here follow a small set of stable
 *   shapes and we only need names, kinds, and preceding JSDoc. An AST dep
 *   (typescript, svelte/compiler) would be heavier than the task warrants.
 * - Re-runs on prebuild so committed MDX is always consistent with source. The
 *   files ARE committed — fresh clones run without needing a prebuild step.
 * - Svelte components: when .svelte files land (Wave B+), we pull JSDoc from
 *   the `<script lang="ts">` block and the `interface Props` declaration.
 */
import { readFile, writeFile, mkdir, rm, readdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const DOCS_ROOT = resolve(HERE, '..');
const REPO_ROOT = resolve(DOCS_ROOT, '..', '..');
const LIB_SRC = resolve(REPO_ROOT, 'packages/svelte-mafs/src');
const API_PAGES_DIR = resolve(DOCS_ROOT, 'src/pages/api');
const MANIFEST_PATH = resolve(DOCS_ROOT, 'src/data/api-manifest.json');

interface ExportRecord {
  name: string;
  kind: 'value' | 'type';
  module: string;
  jsdoc?: string;
  signature?: string;
}

interface ModuleDoc {
  module: string;
  file: string;
  description?: string;
  exports: ExportRecord[];
}

const log = (msg: string) => console.log(`[docgen] ${msg}`);

/** Parse re-export lines from index.ts. Supports the patterns Stream 1 uses. */
function parseIndexExports(indexSrc: string): ExportRecord[] {
  const out: ExportRecord[] = [];
  // Matches: `export { a, b, c } from "./module.js";`
  // and:     `export type { A, B } from "./module.js";`
  const re = /export\s+(type\s+)?\{([^}]+)\}\s+from\s+["']\.\/([^"']+?)\.js["']/g;
  for (const match of indexSrc.matchAll(re)) {
    const isType = Boolean(match[1]);
    const names = match[2]!
      .split(',')
      .map((s) => s.trim().replace(/^type\s+/, ''))
      .filter(Boolean);
    const module = match[3]!;
    for (const name of names) {
      out.push({ name, kind: isType ? 'type' : 'value', module });
    }
  }
  return out;
}

/** Extract JSDoc + signature for a named export from a TS source. */
function extractFromTs(
  src: string,
  name: string,
): { jsdoc?: string; signature?: string } {
  const headerRe = new RegExp(
    String.raw`export\s+(?:async\s+)?(const|function|let|var|type|interface|class)\s+${name}\b`,
    'm',
  );
  const headerHit = src.match(headerRe);
  if (!headerHit || headerHit.index == null) return {};

  const signature = extractSignatureFrom(src, headerHit.index, headerHit[1]!);
  const jsdocRaw = findPrecedingJsDoc(src, headerHit.index);
  const jsdoc = jsdocRaw ? cleanJsDoc(jsdocRaw) : undefined;

  return { jsdoc, signature };
}

/** Walk backwards from `offset` past whitespace to find the immediately-preceding JSDoc block. */
function findPrecedingJsDoc(src: string, offset: number): string | undefined {
  let i = offset - 1;
  while (i >= 0 && /\s/.test(src[i]!)) i--;
  if (i < 1 || src[i] !== '/' || src[i - 1] !== '*') return;
  const closeEnd = i + 1;
  // Walk backwards to the matching `/**`
  let j = i - 2;
  while (j >= 2) {
    if (src[j] === '*' && src[j - 1] === '*' && src[j - 2] === '/') {
      return src.slice(j - 2, closeEnd);
    }
    // A `*/` before our `/**` would mean this isn't a contiguous block
    if (src[j] === '/' && src[j - 1] === '*') return;
    j--;
  }
  return;
}

/** Read a declaration starting at `offset` until the body or statement-end. */
function extractSignatureFrom(src: string, offset: number, kind: string): string {
  if (kind === 'type') {
    const end = src.indexOf(';', offset);
    return dedent(src.slice(offset, end === -1 ? src.length : end + 1));
  }
  if (kind === 'interface' || kind === 'class') {
    const brace = src.indexOf('{', offset);
    if (brace === -1) return dedent(src.slice(offset, src.indexOf('\n', offset)));
    let depth = 1;
    let i = brace + 1;
    while (i < src.length && depth > 0) {
      const c = src[i];
      if (c === '{') depth++;
      else if (c === '}') depth--;
      i++;
    }
    return dedent(src.slice(offset, i));
  }
  // function / const / let / var — stop at `=>`, `{` body, or `;`, whichever first
  let parens = 0;
  let braces = 0;
  for (let i = offset; i < src.length; i++) {
    const c = src[i];
    if (c === '(') parens++;
    else if (c === ')') parens--;
    else if (c === '<') braces++;
    else if (c === '>') braces--;
    if (parens === 0 && braces <= 0) {
      if (c === '=' && src[i + 1] === '>') return dedent(src.slice(offset, i + 2)) + ' …';
      if (c === '{') return dedent(src.slice(offset, i)).trim();
      if (c === ';') return dedent(src.slice(offset, i + 1));
    }
  }
  return dedent(src.slice(offset));
}

/** Strip common leading indent across lines so signature renders cleanly in MDX. */
function dedent(block: string): string {
  const lines = block.split('\n');
  const indent = lines
    .filter((l) => l.trim().length > 0)
    .map((l) => l.match(/^\s*/)![0].length)
    .reduce((a, b) => Math.min(a, b), Infinity);
  const min = Number.isFinite(indent) ? indent : 0;
  return lines.map((l) => l.slice(min)).join('\n').trim();
}

/** Extract module-level "header" JSDoc — the comment at the top of the file. */
function extractModuleDescription(src: string): string | undefined {
  const hit = src.match(/^\s*\/\*\*([\s\S]*?)\*\//);
  return hit ? cleanJsDoc(`/**${hit[1]}*/`) : undefined;
}

function cleanJsDoc(raw: string): string {
  return raw
    .replace(/^\s*\/\*\*\s*/, '')
    .replace(/\s*\*\/\s*$/, '')
    .split('\n')
    .map((line) => line.replace(/^\s*\*\s?/, ''))
    .join('\n')
    .trim();
}

async function loadModuleDoc(mod: string, exports: ExportRecord[]): Promise<ModuleDoc | null> {
  const tsPath = join(LIB_SRC, `${mod}.ts`);
  const sveltePath = join(LIB_SRC, `${mod}.svelte`);
  let src: string;
  let file: string;
  if (existsSync(tsPath)) {
    src = await readFile(tsPath, 'utf8');
    file = tsPath;
  } else if (existsSync(sveltePath)) {
    src = await readFile(sveltePath, 'utf8');
    file = sveltePath;
  } else {
    log(`SKIP ${mod}: no source found (expected ${mod}.ts or ${mod}.svelte)`);
    return null;
  }

  const description = extractModuleDescription(src);
  for (const rec of exports) {
    const parsed = extractFromTs(src, rec.name);
    rec.jsdoc = parsed.jsdoc;
    rec.signature = parsed.signature;
  }
  return { module: mod, file: relative(REPO_ROOT, file), description, exports };
}

function renderMdx(doc: ModuleDoc): string {
  const lines: string[] = [];
  lines.push('---');
  lines.push('layout: ../../layouts/Base.astro');
  lines.push(`title: ${JSON.stringify(doc.module)}`);
  lines.push(`description: ${JSON.stringify(`API reference for the ${doc.module} module of svelte-mafs.`)}`);
  lines.push('prose: true');
  lines.push('---');
  lines.push('');
  lines.push(`# \`${doc.module}\``);
  lines.push('');
  lines.push(`<p class="source-link">Source: <a href="https://github.com/SkipSupreme/svelte-mafs/blob/main/${doc.file}"><code>${doc.file}</code></a></p>`);
  lines.push('');
  if (doc.description) {
    lines.push(doc.description);
    lines.push('');
  }

  const values = doc.exports.filter((e) => e.kind === 'value');
  const types = doc.exports.filter((e) => e.kind === 'type');

  if (types.length) {
    lines.push('## Types');
    lines.push('');
    for (const t of types) {
      lines.push(`### \`${t.name}\``);
      lines.push('');
      if (t.signature) {
        lines.push('```ts');
        lines.push(t.signature);
        lines.push('```');
        lines.push('');
      }
      if (t.jsdoc) {
        lines.push(t.jsdoc);
        lines.push('');
      }
    }
  }

  if (values.length) {
    lines.push('## Exports');
    lines.push('');
    for (const v of values) {
      lines.push(`### \`${v.name}\``);
      lines.push('');
      if (v.signature) {
        lines.push('```ts');
        lines.push(v.signature);
        lines.push('```');
        lines.push('');
      }
      if (v.jsdoc) {
        lines.push(v.jsdoc);
        lines.push('');
      }
    }
  }

  return lines.join('\n');
}

async function clearGenerated(): Promise<void> {
  if (!existsSync(API_PAGES_DIR)) return;
  const files = await readdir(API_PAGES_DIR);
  for (const f of files) {
    if (f.endsWith('.mdx')) {
      await rm(join(API_PAGES_DIR, f));
    }
  }
}

async function main(): Promise<void> {
  const indexPath = join(LIB_SRC, 'index.ts');
  if (!existsSync(indexPath)) {
    log(`no lib source at ${indexPath} — skipping`);
    return;
  }
  const indexSrc = await readFile(indexPath, 'utf8');
  const flat = parseIndexExports(indexSrc);

  if (flat.length === 0) {
    log('no re-exports found; emitting empty manifest');
  }

  const byModule = new Map<string, ExportRecord[]>();
  for (const rec of flat) {
    const arr = byModule.get(rec.module) ?? [];
    arr.push(rec);
    byModule.set(rec.module, arr);
  }

  const docs: ModuleDoc[] = [];
  for (const [mod, exports] of byModule) {
    const doc = await loadModuleDoc(mod, exports);
    if (doc) docs.push(doc);
  }

  await mkdir(API_PAGES_DIR, { recursive: true });
  await clearGenerated();
  for (const doc of docs) {
    const out = join(API_PAGES_DIR, `${doc.module}.mdx`);
    await mkdir(dirname(out), { recursive: true });
    await writeFile(out, renderMdx(doc), 'utf8');
    log(`wrote ${relative(DOCS_ROOT, out)} (${doc.exports.length} exports)`);
  }

  const manifest = {
    modules: docs.map((d) => ({
      module: d.module,
      description: d.description?.split('\n')[0] ?? null,
      exports: d.exports.map((e) => ({ name: e.name, kind: e.kind })),
    })),
  };
  await mkdir(dirname(MANIFEST_PATH), { recursive: true });
  await writeFile(MANIFEST_PATH, JSON.stringify(manifest, null, 2) + '\n', 'utf8');
  log(`wrote ${relative(DOCS_ROOT, MANIFEST_PATH)} (${docs.length} modules)`);
}

main().catch((err) => {
  console.error('[docgen] failed:', err);
  process.exit(1);
});
