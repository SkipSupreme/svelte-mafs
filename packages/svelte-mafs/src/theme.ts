/**
 * Theme tokens exposed as JS strings that resolve to CSS custom properties.
 *
 * Components should reach for `colors.blue` rather than hard-coding `#3b82f6`
 * so `core.css`'s light/dark variants automatically apply. Consumers can
 * override any token at the `.mafs-root` layer without touching components.
 */

const cssVar = (name: string): string => `var(--mafs-${name})`;

export const colors = {
  fg: cssVar("fg"),
  bg: cssVar("bg"),
  line: cssVar("line-color"),
  grid: cssVar("grid-color"),
  blue: cssVar("blue"),
  red: cssVar("red"),
  green: cssVar("green"),
  yellow: cssVar("yellow"),
  orange: cssVar("orange"),
  purple: cssVar("purple"),
  pink: cssVar("pink"),
} as const;

export type ThemeColor = keyof typeof colors;

export const fonts = {
  numeric: cssVar("font-numeric"),
} as const;

export const theme = {
  colors,
  fonts,
} as const;
