import { beforeEach, describe, expect, it, vi } from "vitest";

// Stub Svelte's context API with a shared in-memory map. This mirrors the
// real contract (set-then-get round-trips by symbol key) without needing a
// component mount, which the current test config can't support uniformly.
const contextStore = new Map<symbol, unknown>();

vi.mock("svelte", () => ({
  setContext: (key: symbol, value: unknown) => {
    contextStore.set(key, value);
    return value;
  },
  getContext: (key: symbol) => contextStore.get(key),
}));

beforeEach(() => {
  contextStore.clear();
});

// Import AFTER the mock so pane-context picks up the stubbed svelte module.
const { setPaneContext, getPaneContext } = await import("./pane-context.js");
type PaneContext = Awaited<
  ReturnType<typeof setPaneContext>
>;

describe("pane-context", () => {
  it("returns undefined when no pane has been set", () => {
    expect(getPaneContext()).toBeUndefined();
  });

  it("round-trips the same object after setPaneContext", () => {
    const ctx: PaneContext = { viewBox: { xMin: 0, xMax: 10, yMin: 0, yMax: 5 } };
    setPaneContext(ctx);
    expect(getPaneContext()).toBe(ctx);
  });

  it("preserves viewBox values through the round-trip", () => {
    setPaneContext({ viewBox: { xMin: -3, xMax: 3, yMin: -2, yMax: 8 } });
    expect(getPaneContext()?.viewBox).toEqual({
      xMin: -3,
      xMax: 3,
      yMin: -2,
      yMax: 8,
    });
  });

  it("setPaneContext returns the context it was given", () => {
    const ctx: PaneContext = { viewBox: { xMin: 0, xMax: 1, yMin: 0, yMax: 1 } };
    expect(setPaneContext(ctx)).toBe(ctx);
  });

  it("later setPaneContext overrides earlier one", () => {
    const first: PaneContext = {
      viewBox: { xMin: 0, xMax: 1, yMin: 0, yMax: 1 },
    };
    const second: PaneContext = {
      viewBox: { xMin: 10, xMax: 20, yMin: 10, yMax: 20 },
    };
    setPaneContext(first);
    setPaneContext(second);
    expect(getPaneContext()).toBe(second);
  });
});
