// Matmul with the right-hand operand transposed. C = A · Bᵀ.
//   A: [M, K] row-major.
//   B: [N, K] row-major (i.e. logically [K, N] then transposed → store rows of N indexed by k).
//   C: [M, N] row-major.
// Backs unembedding: A = hidden states [B·T, d], B = wte [V, d] (tied embedding),
// C = logits [B·T, V] computed as logits[i, v] = Σ_k h[i, k] · wte[v, k].

const TILE: u32 = 16u;

@group(0) @binding(0) var<storage, read>          A:    array<f32>;
@group(0) @binding(1) var<storage, read>          B:    array<f32>;
@group(0) @binding(2) var<storage, read_write>    C:    array<f32>;
@group(0) @binding(3) var<uniform>                dims: vec4<u32>; // (M, K, N, _)

var<workgroup> aTile: array<f32, 256>;
var<workgroup> bTile: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(
  @builtin(workgroup_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let M = dims.x; let K = dims.y; let N = dims.z;
  let row = gid.y * TILE + lid.y; // index into M
  let col = gid.x * TILE + lid.x; // index into N

  var sum: f32 = 0.0;
  let nTiles = (K + TILE - 1u) / TILE;

  for (var t: u32 = 0u; t < nTiles; t = t + 1u) {
    let kA = t * TILE + lid.x;          // A is read at [row, kA]
    let kB = t * TILE + lid.x;          // B is read at [col, kB] (note: same lid.x both halves)
    let aIdx = lid.y * TILE + lid.x;
    if (row < M && kA < K) { aTile[aIdx] = A[row * K + kA]; } else { aTile[aIdx] = 0.0; }
    // For Bᵀ we want bTile[t*TILE+r, c] for r in workgroup: store B[col_c, kB] at [r=lid.y, c=lid.x] of the tile ...
    // Simpler: load b chunk indexed by (col_owned_by_lid.y_offset?) – we instead lay out bTile so the inner loop reads column-major.
    // Use mapping: bTile[k*TILE + c] := B[(gid.x*TILE + c), (t*TILE + k)].
    let cInTile = lid.x;
    let kInTile = lid.y;
    let nLocal = gid.x * TILE + cInTile;
    let kLocal = t * TILE + kInTile;
    if (nLocal < N && kLocal < K) {
      bTile[kInTile * TILE + cInTile] = B[nLocal * K + kLocal];
    } else {
      bTile[kInTile * TILE + cInTile] = 0.0;
    }
    workgroupBarrier();

    for (var k: u32 = 0u; k < TILE; k = k + 1u) {
      sum = sum + aTile[lid.y * TILE + k] * bTile[k * TILE + lid.x];
    }
    workgroupBarrier();
  }

  if (row < M && col < N) { C[row * N + col] = sum; }
}
