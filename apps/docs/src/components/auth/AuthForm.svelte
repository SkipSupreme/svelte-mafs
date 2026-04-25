<script lang="ts">
  type Mode = 'signin' | 'signup';

  let { mode = 'signin' as Mode, error = '' } = $props<{ mode?: Mode; error?: string }>();

  let email = $state('');
  let status = $state<'idle' | 'sending' | 'sent' | 'error'>(error ? 'error' : 'idle');
  let errorMsg = $state(error);

  async function submitEmail(ev: SubmitEvent) {
    ev.preventDefault();
    status = 'sending';
    errorMsg = '';
    try {
      const res = await fetch('/api/auth/sign-in/magic-link', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ email, callbackURL: '/welcome' }),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || 'Could not send sign-in link');
      }
      status = 'sent';
    } catch (e) {
      status = 'error';
      errorMsg = e instanceof Error ? e.message : 'Something went wrong';
    }
  }
</script>

<section class="auth-card">
  <h1>{mode === 'signup' ? 'Sign up for Tinker' : 'Welcome back'}</h1>
  <p class="lede">
    {mode === 'signup'
      ? 'Tinker is in alpha. One course is live; new modules every week. Free.'
      : 'Sign in with email or a connected provider.'}
  </p>

  <div class="providers">
    <a class="btn provider" href="/api/auth/sign-in/social?provider=google&callbackURL=/welcome">
      <span class="icon" aria-hidden="true">G</span>
      Continue with Google
    </a>
    <a class="btn provider" href="/api/auth/sign-in/social?provider=github&callbackURL=/welcome">
      <span class="icon" aria-hidden="true">⌥</span>
      Continue with GitHub
    </a>
  </div>

  <div class="sep" role="separator" aria-orientation="horizontal"><span>or with email</span></div>

  {#if status === 'sent'}
    <div class="sent">
      <h2>Check your inbox</h2>
      <p>
        We sent a sign-in link to <strong>{email}</strong>. The link is good for 15 minutes
        and can only be used once.
      </p>
      <p class="lede">Didn't get it? <button class="linkish" onclick={() => (status = 'idle')}>Try again</button>.</p>
    </div>
  {:else}
    <form onsubmit={submitEmail} novalidate>
      <label for="auth-email" class="sr-only">Email address</label>
      <input
        id="auth-email"
        type="email"
        autocomplete="email"
        required
        bind:value={email}
        placeholder="you@example.com"
        disabled={status === 'sending'}
      />
      <button type="submit" disabled={status === 'sending' || !email}>
        {status === 'sending' ? 'Sending…' : 'Send sign-in link'}
      </button>
      {#if status === 'error' && errorMsg}
        <p class="err" role="alert">{errorMsg}</p>
      {/if}
    </form>
  {/if}

  {#if mode === 'signup'}
    <p class="alt">Already have an account? <a href="/signin">Sign in</a></p>
  {:else}
    <p class="alt">New to Tinker? <a href="/signup">Sign up</a></p>
  {/if}
</section>

<style>
  .auth-card {
    max-width: 420px;
    margin: 4rem auto;
    padding: 2rem 1.5rem;
    background: var(--site-bg);
    border: 1px solid var(--site-border);
    border-radius: var(--radius-lg);
  }
  h1 {
    font-family: var(--font-display);
    font-size: 1.75rem;
    font-weight: 600;
    letter-spacing: -0.01em;
    margin: 0 0 0.5rem;
  }
  h2 {
    font-size: 1.125rem;
    font-weight: 600;
    margin: 0 0 0.5rem;
  }
  .lede {
    color: var(--site-fg-muted);
    margin: 0 0 1.5rem;
    font-size: 0.95rem;
  }
  .providers {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    margin-bottom: 1.25rem;
  }
  .btn.provider {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.6rem;
    padding: 0.7rem 1rem;
    border: 1px solid var(--site-border);
    border-radius: var(--radius-md);
    color: var(--site-fg);
    text-decoration: none;
    font-weight: 500;
    background: var(--site-surface);
    transition: background-color 120ms ease, border-color 120ms ease;
  }
  .btn.provider:hover { background: color-mix(in srgb, var(--site-surface) 70%, var(--site-bg)); }
  .icon {
    display: inline-grid;
    place-items: center;
    width: 1.25rem;
    height: 1.25rem;
    background: var(--site-bg);
    border: 1px solid var(--site-border);
    border-radius: var(--radius-sm);
    font-size: 0.75rem;
    font-family: var(--font-mono);
  }
  .sep {
    display: flex;
    align-items: center;
    margin: 1rem 0;
    color: var(--site-fg-muted);
    font-size: 0.85rem;
  }
  .sep::before, .sep::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--site-border);
  }
  .sep span { padding: 0 0.75rem; }
  form { display: flex; flex-direction: column; gap: 0.5rem; }
  input {
    width: 100%;
    padding: 0.7rem 0.85rem;
    border: 1px solid var(--site-border);
    border-radius: var(--radius-md);
    background: var(--site-bg);
    color: var(--site-fg);
    font-size: 1rem;
    font-family: inherit;
  }
  input:focus { outline: 2px solid var(--site-focus); outline-offset: 2px; }
  button[type="submit"] {
    padding: 0.7rem 1rem;
    border: 1px solid var(--site-fg);
    border-radius: var(--radius-md);
    background: var(--site-fg);
    color: var(--site-bg);
    font-weight: 600;
    cursor: pointer;
    transition: opacity 120ms ease, transform 120ms ease;
  }
  button[type="submit"]:disabled { opacity: 0.5; cursor: not-allowed; }
  button[type="submit"]:not(:disabled):hover { transform: translateY(-1px); }
  .err {
    color: #b91c1c;
    font-size: 0.9rem;
    margin: 0;
  }
  .alt {
    color: var(--site-fg-muted);
    font-size: 0.9rem;
    margin: 1.5rem 0 0;
    text-align: center;
  }
  .alt a { color: var(--site-fg); }
  .sent { text-align: left; }
  .linkish {
    background: none;
    border: none;
    color: var(--site-fg);
    text-decoration: underline;
    cursor: pointer;
    padding: 0;
    font: inherit;
  }
  .sr-only {
    position: absolute;
    width: 1px; height: 1px; padding: 0; margin: -1px;
    overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0;
  }
</style>
