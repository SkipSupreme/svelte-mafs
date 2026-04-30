// embeddingGather: out[i, k] = wte[tokIds[i], k] + wpe[(i % T), k]
// Dispatch: ceil((rows*d) / 64). Each thread writes one (row, k) cell.

@group(0) @binding(0) var<storage, read>          tokIds: array<i32>;
@group(0) @binding(1) var<storage, read>          wte:    array<f32>; // [V, d]
@group(0) @binding(2) var<storage, read>          wpe:    array<f32>; // [T, d]
@group(0) @binding(3) var<storage, read_write>    out:    array<f32>; // [rows, d]
@group(0) @binding(4) var<uniform>                dims:   vec4<u32>;  // (rows, d, T, _)

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let rows = dims.x; let d = dims.y; let T = dims.z;
  if (i >= rows * d) { return; }
  let row = i / d;
  let k   = i % d;
  let tok = u32(tokIds[row]);
  let pos = row % T;
  out[i] = wte[tok * d + k] + wpe[pos * d + k];
}
