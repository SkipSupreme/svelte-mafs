<script lang="ts">
  type Mode = 'signin' | 'signup';
  type Provider = 'google' | 'github';
  type View = 'password' | 'magic';

  let {
    mode = 'signin' as Mode,
    error = '',
    providers = [] as Provider[],
    callbackURL = '/welcome',
  } = $props<{
    mode?: Mode;
    error?: string;
    providers?: Provider[];
    callbackURL?: string;
  }>();

  let view = $state<View>('password');
  let email = $state('');
  let password = $state('');
  let name = $state('');
  let status = $state<'idle' | 'sending' | 'sent' | 'error'>(error ? 'error' : 'idle');
  let errorMsg = $state(error);

  function toggleMode(ev: Event) {
    ev.preventDefault();
    // Real navigation so the URL stays honest and link-shareable.
    const next = mode === 'signin' ? '/auth?mode=signup' : '/auth?mode=signin';
    window.location.assign(next);
  }

  async function submitPassword(ev: SubmitEvent) {
    ev.preventDefault();
    status = 'sending';
    errorMsg = '';
    const endpoint = mode === 'signup' ? '/api/auth/sign-up/email' : '/api/auth/sign-in/email';
    const body =
      mode === 'signup'
        ? { email, password, name: name.trim() || email.split('@')[0], callbackURL }
        : { email, password, callbackURL };
    try {
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        // Map a few common statuses without echoing server body — that can
        // leak Better Auth internals or hint at which emails are registered.
        if (res.status === 401 || res.status === 403) {
          throw new Error('That email and password combination didn’t match.');
        }
        if (res.status === 422 || res.status === 400) {
          throw new Error(
            mode === 'signup'
              ? 'Check your email and pick a password with at least 8 characters.'
              : 'Check your email and password and try again.',
          );
        }
        if (res.status === 409) {
          throw new Error('An account with that email already exists. Try signing in.');
        }
        if (res.status === 429) {
          throw new Error('Too many tries. Wait a minute and try again.');
        }
        throw new Error('Something went wrong. Try again.');
      }
      // Better Auth sets the session cookie via Set-Cookie on this response.
      window.location.assign(callbackURL);
    } catch (e) {
      status = 'error';
      errorMsg = e instanceof Error ? e.message : 'Something went wrong';
    }
  }

  async function submitMagic(ev: SubmitEvent) {
    ev.preventDefault();
    status = 'sending';
    errorMsg = '';
    try {
      const res = await fetch('/api/auth/sign-in/magic-link', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ email, callbackURL }),
      });
      if (!res.ok) {
        if (res.status === 429) {
          throw new Error('Too many sign-in requests. Try again in a minute.');
        }
        if (res.status >= 500) {
          throw new Error('Sign-in email delivery is having issues — try the password form instead.');
        }
        throw new Error('Couldn’t send sign-in link. Try again.');
      }
      status = 'sent';
    } catch (e) {
      status = 'error';
      errorMsg = e instanceof Error ? e.message : 'Something went wrong';
    }
  }

  const isSignup = $derived(mode === 'signup');
  const hasProviders = $derived(providers.length > 0);
</script>

<section class="auth-card">
  <h1>{isSignup ? 'Sign up for Tinker' : 'Welcome back'}</h1>
  <p class="lede">
    {isSignup
      ? 'Tinker is in alpha. One course is live; new modules every week. Free.'
      : 'Sign in to pick up where you left off.'}
  </p>

  {#if hasProviders}
    <div class="providers">
      {#if providers.includes('google')}
        <a class="btn provider" href={`/api/auth/sign-in/social?provider=google&callbackURL=${encodeURIComponent(callbackURL)}`}>
          <span class="icon" aria-hidden="true">G</span>
          Continue with Google
        </a>
      {/if}
      {#if providers.includes('github')}
        <a class="btn provider" href={`/api/auth/sign-in/social?provider=github&callbackURL=${encodeURIComponent(callbackURL)}`}>
          <span class="icon" aria-hidden="true">⌥</span>
          Continue with GitHub
        </a>
      {/if}
    </div>
    <div class="sep" role="separator" aria-orientation="horizontal"><span>or with email</span></div>
  {/if}

  {#if status === 'sent'}
    <div class="sent">
      <h2>Check your inbox</h2>
      <p>
        We sent a sign-in link to <strong>{email}</strong>. The link is good for 15 minutes
        and can only be used once.
      </p>
      <p class="lede">Didn't get it? <button class="linkish" onclick={() => (status = 'idle')}>Try again</button>.</p>
    </div>
  {:else if view === 'password'}
    <form onsubmit={submitPassword} novalidate>
      <label for="auth-email" class="sr-only">Email address</label>
      <input
        id="auth-email"
        type="email"
        autocomplete="email"
        required
        bind:value={email}
        placeholder="Email"
        disabled={status === 'sending'}
      />
      <label for="auth-password" class="sr-only">Password</label>
      <input
        id="auth-password"
        type="password"
        autocomplete={isSignup ? 'new-password' : 'current-password'}
        required
        minlength={8}
        bind:value={password}
        placeholder="Password"
        disabled={status === 'sending'}
      />
      {#if isSignup}
        <label for="auth-name" class="sr-only">Display name (optional)</label>
        <input
          id="auth-name"
          type="text"
          autocomplete="nickname"
          bind:value={name}
          placeholder="Display name (optional)"
          disabled={status === 'sending'}
        />
      {/if}
      <button type="submit" disabled={status === 'sending' || !email || !password}>
        {#if status === 'sending'}
          {isSignup ? 'Creating account…' : 'Signing in…'}
        {:else if isSignup}
          Create account
        {:else}
          Sign in
        {/if}
      </button>
      {#if status === 'error' && errorMsg}
        <p class="err" role="alert">{errorMsg}</p>
      {/if}
    </form>
    <p class="alt-action">
      Prefer a one-time link?
      <button class="linkish" onclick={() => { view = 'magic'; status = 'idle'; errorMsg = ''; }}>
        Email me a sign-in link
      </button>
    </p>
  {:else}
    <form onsubmit={submitMagic} novalidate>
      <label for="auth-email-magic" class="sr-only">Email address</label>
      <input
        id="auth-email-magic"
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
    <p class="alt-action">
      <button class="linkish" onclick={() => { view = 'password'; status = 'idle'; errorMsg = ''; }}>
        Use a password instead
      </button>
    </p>
  {/if}

  <p class="alt">
    {#if isSignup}
      Already have an account? <a href="/signin" onclick={toggleMode}>Sign in</a>
    {:else}
      New to Tinker? <a href="/signup" onclick={toggleMode}>Sign up</a>
    {/if}
  </p>
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
    color: var(--site-error);
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
  .alt-action {
    color: var(--site-fg-muted);
    font-size: 0.85rem;
    margin: 0.85rem 0 0;
    text-align: center;
  }
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
