import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const readCss = (name: string) =>
  readFileSync(resolve(__dirname, name), "utf8");

describe("core.css", () => {
  const css = readCss("./core.css");

  const tokens = [
    "--mafs-fg",
    "--mafs-bg",
    "--mafs-line-color",
    "--mafs-grid-color",
    "--mafs-blue",
    "--mafs-red",
    "--mafs-green",
    "--mafs-yellow",
    "--mafs-orange",
    "--mafs-purple",
    "--mafs-pink",
  ];

  it.each(tokens)("defines %s", (token) => {
    // Each token must be set at least once in a :root-ish rule (light theme).
    expect(css).toMatch(new RegExp(`${token}\\s*:`));
  });

  it("defines a dark variant under [data-theme='dark'] .mafs-root", () => {
    // Plan-mandated selector. Use :where() so specificity stays 0 and
    // consumer overrides win without !important.
    expect(css).toMatch(/:where\(\[data-theme=["']dark["']\]\)\s+\.mafs-root/);
  });

  it("redefines every token in the dark variant", () => {
    // Find the dark block and assert each token reappears inside it.
    // The dark selector may be part of a selector list (e.g. accept both
    // `.mafs-root` and `[data-mafs-root]`), so the regex allows any
    // characters up to the opening brace.
    const darkBlock = css.match(
      /:where\(\[data-theme=["']dark["']\]\)\s+\.mafs-root[^{]*\{([^}]*)\}/,
    );
    expect(darkBlock, "dark variant block missing").toBeTruthy();
    const body = darkBlock![1];
    for (const token of tokens) {
      expect(body).toMatch(new RegExp(`${token}\\s*:`));
    }
  });

  it("matches snapshot", () => {
    expect(css).toMatchSnapshot();
  });
});

describe("font.css", () => {
  const css = readCss("./font.css");

  it("declares a --mafs-font-numeric custom property", () => {
    expect(css).toMatch(/--mafs-font-numeric\s*:/);
  });

  it("has a follow-up note for a bundled font", () => {
    // Plan: "start with system-ui mono fallback; note follow-up to ship a
    // real bundled font". Keep the marker greppable for future work.
    expect(css).toMatch(/TODO.*bundled font/i);
  });
});
