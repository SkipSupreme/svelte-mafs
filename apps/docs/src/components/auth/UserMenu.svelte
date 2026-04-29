<script lang="ts">
  interface User {
    id: string;
    email: string;
    name?: string | null;
    image?: string | null;
    role?: string;
  }

  let user = $state<User | null>(null);
  let loaded = $state(false);
  let open = $state(false);

  async function loadSession() {
    try {
      const res = await fetch('/api/auth/get-session', {
        credentials: 'same-origin',
      });
      if (res.ok) {
        const data = (await res.json()) as { user?: User } | null;
        user = data?.user ?? null;
      }
    } catch {
      user = null;
    } finally {
      loaded = true;
    }
  }

  async function signOut() {
    try {
      await fetch('/api/auth/sign-out', { method: 'POST', credentials: 'same-origin' });
    } finally {
      location.href = '/';
    }
  }

  function handleDocumentClick(ev: MouseEvent) {
    if (!(ev.target instanceof Node)) return;
    const root = document.querySelector('.usermenu-root');
    if (root && !root.contains(ev.target)) open = false;
  }

  $effect(() => {
    loadSession();
    document.addEventListener('click', handleDocumentClick);
    return () => document.removeEventListener('click', handleDocumentClick);
  });

  function initial(u: User) {
    return (u.name ?? u.email)[0]?.toUpperCase() ?? '?';
  }
</script>

<div class="usermenu-root">
  {#if !loaded}
    <span class="placeholder" aria-hidden="true"></span>
  {:else if user}
    <button
      class="avatar-btn"
      type="button"
      aria-haspopup="menu"
      aria-expanded={open}
      onclick={() => (open = !open)}
    >
      {#if user.image}
        <img src={user.image} alt="" />
      {:else}
        <span class="initial">{initial(user)}</span>
      {/if}
    </button>
    {#if open}
      <ul class="menu" role="menu">
        <li role="none" class="email"><span>{user.email}</span></li>
        <li role="none"><a role="menuitem" href="/me">Account</a></li>
        <li role="none"><a role="menuitem" href="/me/bookmarks">Bookmarks</a></li>
        {#if user.role === 'admin'}
          <li role="none"><a role="menuitem" href="/admin">Admin</a></li>
        {/if}
        <li role="none" class="divider" aria-hidden="true"></li>
        <li role="none"><button role="menuitem" type="button" onclick={signOut}>Sign out</button></li>
      </ul>
    {/if}
  {:else}
    <a href="/signin" class="muted">Sign in</a>
    <a href="/signup" class="cta">Sign up</a>
  {/if}
</div>

<style>
  .usermenu-root {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    position: relative;
  }
  .placeholder {
    display: inline-block;
    width: 2rem;
    height: 2rem;
    border-radius: 50%;
    background: var(--site-surface);
  }
  .avatar-btn {
    width: 2rem;
    height: 2rem;
    border-radius: 50%;
    border: 1px solid var(--site-border);
    background: var(--site-surface);
    color: var(--site-fg);
    cursor: pointer;
    display: grid;
    place-items: center;
    overflow: hidden;
    padding: 0;
    font-weight: 600;
    font-size: 0.9rem;
    transition: border-color 120ms ease;
  }
  .avatar-btn:hover { border-color: var(--site-fg-muted); }
  .avatar-btn img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .menu {
    position: absolute;
    top: calc(100% + 0.5rem);
    right: 0;
    min-width: 12rem;
    background: var(--site-bg);
    border: 1px solid var(--site-border);
    border-radius: var(--radius-md);
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    list-style: none;
    padding: 0.4rem;
    margin: 0;
    z-index: 20;
  }
  .menu li { margin: 0; }
  .menu .email {
    padding: 0.4rem 0.6rem 0.6rem;
    font-size: 0.8rem;
    color: var(--site-fg-muted);
    word-break: break-all;
  }
  .menu .divider {
    height: 1px;
    background: var(--site-border);
    margin: 0.25rem 0;
  }
  .menu a, .menu button {
    display: block;
    width: 100%;
    text-align: left;
    padding: 0.45rem 0.6rem;
    border-radius: var(--radius-sm);
    color: var(--site-fg);
    text-decoration: none;
    font-size: 0.9rem;
    background: none;
    border: none;
    cursor: pointer;
    font: inherit;
  }
  .menu a:hover, .menu button:hover { background: var(--site-surface); }
  a.muted { color: var(--site-fg-muted); padding: 0.4rem 0.65rem; font-size: 0.95rem; }
  a.muted:hover { color: var(--site-fg); }
  a.cta {
    background: var(--site-fg);
    color: var(--site-bg);
    padding: 0.45rem 0.85rem;
    border-radius: var(--radius-md);
    font-weight: 600;
    font-size: 0.9rem;
    text-decoration: none;
  }
  a.cta:hover { transform: translateY(-1px); }
</style>
