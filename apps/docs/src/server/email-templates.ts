export const magicLinkTemplate = ({ url }: { url: string }) => `<!doctype html>
<html><body style="font-family:system-ui,-apple-system,sans-serif;max-width:560px;margin:auto;padding:24px;color:#111">
<h2 style="margin:0 0 16px">Sign in to Tinker</h2>
<p>Click the button below to sign in. This link expires in 15 minutes and can only be used once.</p>
<p style="margin:24px 0">
  <a href="${url}" style="display:inline-block;padding:12px 22px;background:#111;color:#fff;border-radius:8px;text-decoration:none;font-weight:600">Sign in to Tinker</a>
</p>
<p style="color:#666;font-size:14px">If the button doesn't work, copy and paste this URL into your browser:<br><span style="word-break:break-all">${url}</span></p>
<p style="color:#999;font-size:13px;margin-top:32px">If you didn't request this, you can safely ignore this email.</p>
</body></html>`;

export const welcomeTemplate = () => `<!doctype html>
<html><body style="font-family:system-ui,-apple-system,sans-serif;max-width:560px;margin:auto;padding:24px;color:#111">
<h2 style="margin:0 0 16px">You're in.</h2>
<p>Tinker is in alpha. One course is live today: <a href="https://learntinker.com/courses/ml-math" style="color:#111">Math for Machine Learning</a>. New modules every week.</p>
<p>You can opt into a heads-up email when new modules ship at <a href="https://learntinker.com/me" style="color:#111">your account page</a>.</p>
<p style="margin-top:32px">— The Tinker team</p>
</body></html>`;

export const dropTemplate = ({ bodyHtml, unsubUrl }: { bodyHtml: string; unsubUrl: string }) => `<!doctype html>
<html><body style="font-family:system-ui,-apple-system,sans-serif;max-width:560px;margin:auto;padding:24px;color:#111">
${bodyHtml}
<hr style="margin:32px 0;border:none;border-top:1px solid #ddd">
<p style="color:#888;font-size:12px">You're getting this because you opted into Tinker module-launch emails. <a href="${unsubUrl}" style="color:#888">Unsubscribe</a>.</p>
</body></html>`;

export const accountDeletedTemplate = () => `<!doctype html>
<html><body style="font-family:system-ui,-apple-system,sans-serif;max-width:560px;margin:auto;padding:24px;color:#111">
<h2 style="margin:0 0 16px">Your Tinker account has been deleted.</h2>
<p>This is confirmation that your account, progress, bookmarks, and notes have been removed. Your email has been anonymized and will no longer receive any messages from Tinker.</p>
<p>If this wasn't you, reply to this email immediately.</p>
</body></html>`;
