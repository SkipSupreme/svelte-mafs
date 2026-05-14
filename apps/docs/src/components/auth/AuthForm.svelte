<script lang="ts">
  type Mode = 'signin' | 'signup';

  let {
    mode = 'signin' as Mode,
    error = '',
    callbackURL = '/welcome',
  } = $props<{ mode?: Mode; error?: string; callbackURL?: string }>();

  let email = $state('');
  let password = $state('');
  let name = $state('');
  let status = $state<'idle' | 'sending' | 'error'>(error ? 'error' : 'idle');
  let errorMsg = $state(error);

  function toggleMode(ev: Event) {
    ev.preventDefault();
    const next = mode === 'signin' ? '/auth?mode=signup' : '/auth?mode=signin';
    window.location.assign(next);
  }

  async function submit(ev: SubmitEvent) {
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
        if (res.status === 401 || res.status === 403) {
          throw new Error('That email and password combination didn’t match.');
        }
        if (res.status === 409) {
          throw new Error('An account with that email already exists. Try signing in.');
        }
        if (res.status === 422 || res.status === 400) {
          throw new Error(
            mode === 'signup'
              ? 'Check your email and pick a password with at least 8 characters.'
              : 'Check your email and password and try again.',
          );
        }
        if (res.status === 429) {
          throw new Error('Too many tries. Wait a minute and try again.');
        }
        throw new Error('Something went wrong. Try again.');
      }
      window.location.assign(callbackURL);
    } catch (e) {
      status = 'error';
      errorMsg = e instanceof Error ? e.message : 'Something went wrong';
    }
  }

  const isSignup = $derived(mode === 'signup');
</script>

<section class="auth-card">
  <h1>{isSignup ? 'Sign up for Tinker' : 'Welcome back'}</h1>
  <p class="lede">
    {isSignup
      ? 'Tinker is in alpha. One course is live; new modules every week. Free.'
      : 'Sign in to pick up where you left off.'}
  </p>

  <form onsubmit={submit} novalidate>
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

  <p class="alt">
    {#if isSignup}
      Already have an account? <a href="/auth?mode=signin" onclick={toggleMode}>Sign in</a>
    {:else}
      New to Tinker? <a href="/auth?mode=signup" onclick={toggleMode}>Sign up</a>
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
  .lede {
    color: var(--site-fg-muted);
    margin: 0 0 1.5rem;
    font-size: 0.95rem;
  }
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
  .sr-only {
    position: absolute;
    width: 1px; height: 1px; padding: 0; margin: -1px;
    overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0;
  }
</style>
