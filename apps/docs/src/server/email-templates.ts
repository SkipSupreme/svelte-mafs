// Email clients don't honor CSS variables, so we keep the brand-light palette
// inlined here. Light values are derived from `apps/docs/src/styles/global.css`
// (--site-bg, --site-fg, --site-fg-muted, --site-border) plus a few darker
// muted shades that read on every major email client. If the brand color
// shifts, update this block and DESIGN.md notes the parity.
const COLORS = {
  fg: '#17181a', // body text — matches --site-fg
  fgMuted: '#585a60', // disclaimers — matches --site-fg-muted
  fgQuiet: '#888888', // footer text
  border: '#e4e4e0', // hairline — matches --site-border
  bg: '#fdfdfc', // body bg — matches --site-bg
  cta: '#17181a', // dark button (high contrast on light email canvas)
  ctaFg: '#fdfdfc',
} as const;

const BODY_STYLE =
  `font-family:system-ui,-apple-system,sans-serif;max-width:560px;margin:auto;padding:24px;color:${COLORS.fg}`;

export const dropTemplate = ({ bodyHtml, unsubUrl }: { bodyHtml: string; unsubUrl: string }) => `<!doctype html>
<html><body style="${BODY_STYLE}">
${bodyHtml}
<hr style="margin:32px 0;border:none;border-top:1px solid ${COLORS.border}">
<p style="color:${COLORS.fgQuiet};font-size:12px">You're getting this because you opted into Tinker module-launch emails. <a href="${unsubUrl}" style="color:${COLORS.fgQuiet}">Unsubscribe</a>.</p>
</body></html>`;

export const accountDeletedTemplate = () => `<!doctype html>
<html><body style="${BODY_STYLE}">
<h2 style="margin:0 0 16px">Your Tinker account has been deleted.</h2>
<p>This is confirmation that your account, progress, bookmarks, and notes have been removed. Your email has been anonymized and will no longer receive any messages from Tinker.</p>
<p>If this wasn't you, reply to this email immediately.</p>
</body></html>`;
