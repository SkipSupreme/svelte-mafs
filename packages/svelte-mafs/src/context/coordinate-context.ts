import { getContext, setContext } from "svelte";
import type { Vec2 } from "../vec.js";

export interface UserViewBox {
  readonly x: readonly [number, number];
  readonly y: readonly [number, number];
}

export interface NormalizedViewBox {
  readonly xMin: number;
  readonly xMax: number;
  readonly yMin: number;
  readonly yMax: number;
}

export interface CoordContext {
  readonly userToPx: (user: Vec2) => Vec2;
  readonly pxToUser: (px: Vec2) => Vec2;
  readonly viewBox: NormalizedViewBox;
  readonly widthPx: number;
  readonly heightPx: number;
}

export const normalizeViewBox = (vb: UserViewBox): NormalizedViewBox => {
  const [x0, x1] = vb.x;
  const [y0, y1] = vb.y;
  return {
    xMin: Math.min(x0, x1),
    xMax: Math.max(x0, x1),
    yMin: Math.min(y0, y1),
    yMax: Math.max(y0, y1),
  };
};

export const makeCoordContext = (
  viewBox: NormalizedViewBox,
  widthPx: number,
  heightPx: number,
): CoordContext => {
  const { xMin, xMax, yMin, yMax } = viewBox;
  const xSpan = xMax - xMin;
  const ySpan = yMax - yMin;

  const userToPx = (user: Vec2): Vec2 => {
    const px = xSpan === 0 ? 0 : ((user[0] - xMin) / xSpan) * widthPx;
    const py = ySpan === 0 ? 0 : ((yMax - user[1]) / ySpan) * heightPx;
    return [px, py];
  };

  const pxToUser = (px: Vec2): Vec2 => {
    const ux = widthPx === 0 ? xMin : xMin + (px[0] / widthPx) * xSpan;
    const uy = heightPx === 0 ? yMax : yMax - (px[1] / heightPx) * ySpan;
    return [ux, uy];
  };

  return { userToPx, pxToUser, viewBox, widthPx, heightPx };
};

const KEY = Symbol("mafs.coord");

export const setCoordContext = (ctx: CoordContext): CoordContext =>
  setContext(KEY, ctx);

export const getCoordContext = (): CoordContext => {
  const ctx = getContext<CoordContext | undefined>(KEY);
  if (!ctx) {
    throw new Error("<Mafs> context missing. Wrap your component in <Mafs>.");
  }
  return ctx;
};
