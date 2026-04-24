import { describe, expect, it } from "vitest";
import { Coordinates, Plot } from "./index.js";

/**
 * Svelte 5 resolves `<Plot.OfX>` as a JS property lookup, so the namespace
 * contract is simply "every key maps to something the Svelte runtime will
 * accept as a component." That's any function (class) — these checks guard
 * against accidental key typos or export regressions in the two namespaces.
 */

describe("namespace modules", () => {
  describe("Plot", () => {
    const expectedKeys = ["OfX", "OfY", "Parametric", "Inequality", "VectorField"] as const;

    for (const key of expectedKeys) {
      it(`Plot.${key} resolves to a component constructor`, () => {
        expect(Plot).toHaveProperty(key);
        expect(typeof Plot[key]).toBe("function");
      });
    }

    it("exposes exactly the expected keys (catches accidental additions)", () => {
      expect(Object.keys(Plot).sort()).toEqual([...expectedKeys].sort());
    });
  });

  describe("Coordinates", () => {
    it("Coordinates.Cartesian resolves to a component constructor", () => {
      expect(Coordinates).toHaveProperty("Cartesian");
      expect(typeof Coordinates.Cartesian).toBe("function");
    });

    it("exposes only Cartesian in v1 (Polar etc. land later without breakage)", () => {
      expect(Object.keys(Coordinates)).toEqual(["Cartesian"]);
    });
  });
});
