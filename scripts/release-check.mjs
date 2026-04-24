#!/usr/bin/env node
// @ts-check
/**
 * Release readiness check for `svelte-mafs`.
 *
 * Run pre-publish (default) or post-pack via the RELEASE_CHECK_MODE env var:
 *
 *   pre   — fail fast on missing files, secret-shaped strings in the source
 *           tree that would end up in the tarball, or invalid package.json.
 *   post  — additionally inspect the packed tarball (`pnpm pack`) for byte
 *           size, type-declaration presence, and unexpected contents.
 *
 * No external runtime deps — pure Node ESM. Runs on Node >= 20.
 *
 * Exit codes: 0 ok, 1 issues found, 2 unexpected error.
 */

import { execFileSync } from "node:child_process";
import { readFileSync, existsSync, readdirSync, statSync } from "node:fs";
import { dirname, resolve, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const REPO_ROOT = resolve(__dirname, "..");
const LIB_ROOT = resolve(REPO_ROOT, "packages/svelte-mafs");
const MODE = /** @type {"pre" | "post"} */ (process.env.RELEASE_CHECK_MODE ?? "pre");
const VERBOSE = process.env.RELEASE_CHECK_VERBOSE === "1";

/** @type {Array<{level: "error" | "warn"; message: string}>} */
const findings = [];
const ok = (msg) => console.log(`  ok    ${msg}`);
const warn = (msg) => findings.push({ level: "warn", message: msg });
const err = (msg) => findings.push({ level: "error", message: msg });
const log = (msg) => console.log(msg);

/* -------------------------------------------------------------------------- */
/* Checks                                                                     */
/* -------------------------------------------------------------------------- */

/** Validate the lib package.json has the fields npm and consumers need. */
function checkPackageManifest() {
  log("\n[1/5] Validating packages/svelte-mafs/package.json");
  const path = join(LIB_ROOT, "package.json");
  if (!existsSync(path)) {
    err(`package.json missing at ${path}`);
    return null;
  }
  const pkg = JSON.parse(readFileSync(path, "utf8"));

  const required = ["name", "version", "license", "type", "exports", "files"];
  for (const field of required) {
    if (!(field in pkg)) err(`package.json missing required field: ${field}`);
    else ok(`field present: ${field}`);
  }

  if (pkg.private === true) err("package.json has private:true — npm will refuse to publish");
  else ok("package is publishable (private:false or unset)");

  // exports map must declare types
  const dotExport = pkg.exports?.["."];
  if (!dotExport?.types) err("exports['.'].types missing — consumers won't get type declarations");
  else ok("exports['.'].types declared");

  // files must include dist
  if (!Array.isArray(pkg.files) || !pkg.files.some((f) => f === "dist" || f.startsWith("dist/"))) {
    err("package.json files[] must include 'dist'");
  } else {
    ok("dist included in files[]");
  }

  // version must be valid semver
  if (!/^\d+\.\d+\.\d+(-[\w.+-]+)?$/.test(pkg.version)) {
    err(`version is not valid semver: ${pkg.version}`);
  } else {
    ok(`version is valid semver: ${pkg.version}`);
  }

  return pkg;
}

/** Make sure dist/ exists and has the entry points the exports map promises. */
function checkDistContents(pkg) {
  log("\n[2/5] Checking dist/ contents");
  const dist = join(LIB_ROOT, "dist");
  if (!existsSync(dist)) {
    if (MODE === "pre") {
      warn("dist/ does not exist yet (build runs after this in the release workflow)");
    } else {
      err("dist/ does not exist post-build — build silently failed?");
    }
    return;
  }

  const expected = collectExportsTargets(pkg.exports ?? {});
  for (const rel of expected) {
    const abs = join(LIB_ROOT, rel);
    if (!existsSync(abs)) err(`exports map promises ${rel} but it is not in dist/`);
    else ok(`dist contains ${rel}`);
  }

  // Total dist size sanity check.
  const totalBytes = walkSize(dist);
  const kib = (totalBytes / 1024).toFixed(1);
  if (totalBytes > 200 * 1024) warn(`dist/ is ${kib} KiB — expected < 200 KiB for v0.1.0`);
  else ok(`dist/ size: ${kib} KiB`);
}

/** Scan source tree for accidentally-committed secrets that would publish. */
function checkSecretLeaks() {
  log("\n[3/5] Scanning for secret-shaped strings in publishable paths");
  const patterns = [
    { name: "GitHub PAT (classic)", re: /\bghp_[A-Za-z0-9]{36}\b/ },
    { name: "GitHub PAT (fine-grained)", re: /\bgithub_pat_[A-Za-z0-9_]{82}\b/ },
    { name: "AWS access key", re: /\b(AKIA|ASIA)[0-9A-Z]{16}\b/ },
    { name: "npm token", re: /\bnpm_[A-Za-z0-9]{36}\b/ },
    { name: "OpenAI key", re: /\bsk-[A-Za-z0-9]{32,}\b/ },
    { name: "Anthropic key", re: /\bsk-ant-[A-Za-z0-9-]{32,}\b/ },
    { name: "private key block", re: /-----BEGIN (RSA|EC|OPENSSH|PGP) PRIVATE KEY-----/ },
  ];
  const scanRoots = [
    join(LIB_ROOT, "src"),
    join(LIB_ROOT, "dist"),
    join(LIB_ROOT, "package.json"),
  ];
  let scanned = 0;
  let leaks = 0;
  for (const root of scanRoots) {
    if (!existsSync(root)) continue;
    walkFiles(root, (path) => {
      // Skip binary-ish files and source maps (false positives possible).
      if (/\.(png|jpg|jpeg|gif|woff2?|ttf|eot|ico|map)$/i.test(path)) return;
      if (path.endsWith(".env") || path.endsWith(".env.example")) {
        warn(`${relative(REPO_ROOT, path)} should not be in publishable paths`);
        return;
      }
      let content;
      try {
        content = readFileSync(path, "utf8");
      } catch {
        return;
      }
      // Skip self-references in this very script.
      if (path === __filename) return;
      scanned++;
      for (const { name, re } of patterns) {
        if (re.test(content)) {
          err(`${name} pattern matched in ${relative(REPO_ROOT, path)}`);
          leaks++;
        }
      }
    });
  }
  if (leaks === 0) ok(`scanned ${scanned} files, no secret patterns found`);
}

/** Verify .changeset/ has at least one pending changeset OR a clean state. */
function checkChangesets() {
  log("\n[4/5] Inspecting .changeset/ state");
  const changesetDir = join(REPO_ROOT, ".changeset");
  if (!existsSync(changesetDir)) {
    err(".changeset/ directory missing — run `pnpm changeset init` first");
    return;
  }
  const files = readdirSync(changesetDir).filter((f) => f.endsWith(".md") && f !== "README.md");
  if (files.length === 0) {
    ok("no pending changesets — release workflow will be a no-op (expected on most pushes)");
  } else {
    ok(`${files.length} pending changeset${files.length === 1 ? "" : "s"}: ${files.join(", ")}`);
  }
}

/** In post mode, also pack the lib and inspect the actual tarball. */
function checkPackedTarball() {
  if (MODE !== "post") {
    log("\n[5/5] Skipping packed-tarball inspection (RELEASE_CHECK_MODE=pre)");
    return;
  }
  log("\n[5/5] Packing tarball for inspection");
  // execFileSync — no shell, args are an array, no injection surface.
  let packOutput;
  try {
    packOutput = execFileSync("pnpm", ["pack", "--pack-destination", "/tmp"], {
      cwd: LIB_ROOT,
      encoding: "utf8",
    });
  } catch (e) {
    err(`pnpm pack failed: ${e instanceof Error ? e.message : e}`);
    return;
  }
  const tarball = packOutput.trim().split("\n").pop();
  if (!tarball || !existsSync(tarball)) {
    err(`pnpm pack did not produce a tarball at ${tarball}`);
    return;
  }
  const sizeKib = (statSync(tarball).size / 1024).toFixed(1);
  ok(`tarball produced: ${tarball} (${sizeKib} KiB)`);

  let listing;
  try {
    listing = execFileSync("tar", ["-tzf", tarball], { encoding: "utf8" });
  } catch (e) {
    err(`tar listing failed: ${e instanceof Error ? e.message : e}`);
    return;
  }
  const contents = listing.split("\n").filter(Boolean);

  const dtsFiles = contents.filter((p) => p.endsWith(".d.ts"));
  if (dtsFiles.length === 0) err("tarball contains no .d.ts files — types will not be available to consumers");
  else ok(`tarball includes ${dtsFiles.length} .d.ts file(s)`);

  const testFiles = contents.filter((p) => /\.test\.(ts|js)$/.test(p));
  if (testFiles.length > 0) {
    warn(`tarball contains ${testFiles.length} test file(s) — adjust 'files' in package.json`);
  }

  if (VERBOSE) {
    log("  tarball contents:");
    for (const path of contents) log(`    ${path}`);
  }
}

/* -------------------------------------------------------------------------- */
/* Helpers                                                                    */
/* -------------------------------------------------------------------------- */

function collectExportsTargets(exports) {
  const targets = new Set();
  const walk = (value) => {
    if (typeof value === "string") {
      if (value.startsWith("./")) targets.add(value.slice(2));
    } else if (value && typeof value === "object") {
      for (const v of Object.values(value)) walk(v);
    }
  };
  walk(exports);
  return [...targets];
}

function walkFiles(root, fn) {
  const stat = statSync(root);
  if (stat.isFile()) {
    fn(root);
    return;
  }
  for (const entry of readdirSync(root)) {
    if (entry === "node_modules" || entry === ".git" || entry === "coverage") continue;
    walkFiles(join(root, entry), fn);
  }
}

function walkSize(root) {
  let total = 0;
  walkFiles(root, (p) => {
    total += statSync(p).size;
  });
  return total;
}

/* -------------------------------------------------------------------------- */
/* Main                                                                       */
/* -------------------------------------------------------------------------- */

log(`svelte-mafs release-check (mode=${MODE})`);
log(`repo:    ${REPO_ROOT}`);
log(`lib:     ${LIB_ROOT}`);

try {
  const pkg = checkPackageManifest();
  if (pkg) checkDistContents(pkg);
  checkSecretLeaks();
  checkChangesets();
  checkPackedTarball();
} catch (e) {
  console.error("\nUnexpected failure during release-check:", e);
  process.exit(2);
}

const errors = findings.filter((f) => f.level === "error");
const warnings = findings.filter((f) => f.level === "warn");

log("\n----------------------------------------");
if (errors.length === 0 && warnings.length === 0) {
  log("Release check: PASS (0 issues)");
  process.exit(0);
}
for (const w of warnings) console.warn(`  warn  ${w.message}`);
for (const e of errors) console.error(`  ERROR ${e.message}`);
log("----------------------------------------");
log(
  `Release check: ${errors.length === 0 ? "PASS with warnings" : "FAIL"} (${errors.length} error(s), ${warnings.length} warning(s))`,
);
process.exit(errors.length === 0 ? 0 : 1);
