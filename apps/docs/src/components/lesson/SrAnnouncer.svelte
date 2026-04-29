<script lang="ts">
  /**
   * SrAnnouncer — single aria-live="polite" mirror for the alive layer.
   * Listens for `tinker:announce` window events. Dropped into Lesson.astro
   * once; every celebration calls announce(msg) from lib/celebrate.ts.
   */
  import { TINKER_EVENT } from '../../lib/events';

  let message = $state('');

  $effect(() => {
    const onAnnounce = (e: CustomEvent<{ message: string }>) => {
      message = '';
      queueMicrotask(() => {
        message = e.detail.message;
      });
    };
    window.addEventListener(TINKER_EVENT.announce, onAnnounce as EventListener);
    return () =>
      window.removeEventListener(TINKER_EVENT.announce, onAnnounce as EventListener);
  });
</script>

<div role="status" aria-live="polite" aria-atomic="true" class="sr-only">
  {message}
</div>

<style>
  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }
</style>
