/**
 * Stylelint config for Tinker (CQ-1 of /plan-eng-review).
 *
 * Enforces the Quality Bar's "no hex literals in component CSS" rule.
 * Token-definition files (apps/docs/src/styles/*.css) are exempted —
 * named CSS variables MUST be defined somewhere typed.
 *
 * Run: `pnpm lint:css` (workspace-wide) or `pnpm -F docs lint:css` (per-package).
 */
module.exports = {
  extends: ["stylelint-config-standard"],
  rules: {
    // The whole point of this config — block hex literals in components.
    // DESIGN.md is the contract; global.css is the typed implementation;
    // components reference tokens. Anything else is a bug.
    "color-no-hex": true,

    // Below: relax noisy stylistic rules that fight Tailwind / Astro / Svelte
    // conventions. We're enforcing tokens and CORRECTNESS, not formatting.
    "selector-class-pattern": null,
    "no-descending-specificity": null,
    "custom-property-pattern": null,
    "custom-property-empty-line-before": null,
    "keyframes-name-pattern": null,
    "declaration-empty-line-before": null,
    "rule-empty-line-before": null,
    "comment-empty-line-before": null,
    "at-rule-empty-line-before": null,
    "declaration-block-single-line-max-declarations": null,
    "alpha-value-notation": null,
    "color-function-notation": null,
    "media-feature-range-notation": null,
    "selector-not-notation": null,
    "font-family-name-quotes": null,
    "shorthand-property-no-redundant-values": null,
    "no-duplicate-selectors": null,
    "value-keyword-case": null,
    "declaration-block-no-redundant-longhand-properties": null,
    "property-no-vendor-prefix": null,
    "value-no-vendor-prefix": null,
    "selector-pseudo-class-no-unknown": [true, {
      ignorePseudoClasses: ["global", "local", "deep"],
    }],
    "selector-pseudo-element-no-unknown": [true, {
      ignorePseudoElements: ["global", "local", "deep"],
    }],
    "no-empty-source": null,
    "block-no-empty": null,
    "function-no-unknown": null,
    "import-notation": null,
    "string-no-newline": null,
    "no-invalid-double-slash-comments": null,

    // Inline @ rules used by Astro / Svelte build pipelines.
    "at-rule-no-unknown": [true, {
      ignoreAtRules: [
        "tailwind", "apply", "variants", "responsive", "screen",
        "layer", "include", "function", "if", "else", "each",
      ],
    }],
  },
  overrides: [
    // Token-definition files: hex literals are allowed (this is where named
    // tokens are minted). DESIGN.md describes intent in prose; this file is
    // the typed implementation.
    {
      files: ["**/styles/global.css", "**/styles/tokens.css"],
      rules: {
        "color-no-hex": null,
      },
    },
    // Astro and Svelte components: lint <style> blocks the same way.
    {
      files: ["**/*.astro", "**/*.svelte"],
      customSyntax: "postcss-html",
    },
  ],
};
