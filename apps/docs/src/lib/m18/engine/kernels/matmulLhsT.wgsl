// Matmul with the left-hand operand transposed. C = Aᵀ · B.
//   A: [M, K] row-major (logically transposed → produces [K, M] then × B[M, N]).
//   B: [M, N] row-major.
//   C: [K, N] row-major.
// C[k, n] = Σ_m A[m, k] · B[m, n]. Backs matmulBwd_dB: given forward A and
// upstream gradient dC, compute dB = Aᵀ · dC.

const TILE: u32 = 16u;

@group(0) @binding(0) var<storage, read>          A:    array<f32>;
@group(0) @binding(1) var<storage, read>          B:    array<f32>;
@group(0) @binding(2) var<storage, read_write>    C:    array<f32>;
@group(0) @binding(3) var<uniform>                dims: vec4<u32>; // (M, K, N, _)

var<workgroup> aTile: array<f32, 256>; // TILE*TILE — A elements indexed by (m_in_tile, k_in_tile)
var<workgroup> bTile: array<f32, 256>; // B elements indexed by (m_in_tile, n_in_tile)

@compute @workgroup_size(16, 16, 1)
fn main(
  @builtin(workgroup_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let M = dims.x; let K = dims.y; let N = dims.z;
  let kOut = gid.y * TILE + lid.y;  // index into K (rows of C)
  let nOut = gid.x * TILE + lid.x;  // index into N (cols of C)

  var sum: f32 = 0.0;
  let mTiles = (M + TILE - 1u) / TILE;

  for (var t: u32 = 0u; t < mTiles; t = t + 1u) {
    // Load A[mLocal, kOut] into aTile[mInTile, kInTile=lid.y].
    // Each thread loads one A and one B element so the inner loop sees a full tile.
    let mLocal = t * TILE + lid.x;     // m index for the A load
    let aIdx = lid.x * TILE + lid.y;   // store transposed: aTile[m, k]
    if (mLocal < M && kOut < K) {
      aTile[aIdx] = A[mLocal * K + kOut];
    } else {
      aTile[aIdx] = 0.0;
    }
    // Load B[mLocal, nOut] into bTile[mInTile=lid.y, nInTile=lid.x].
    let mLocalB = t * TILE + lid.y;
    let bIdx = lid.y * TILE + lid.x;
    if (mLocalB < M && nOut < N) {
      bTile[bIdx] = B[mLocalB * N + nOut];
    } else {
      bTile[bIdx] = 0.0;
    }
    workgroupBarrier();

    for (var m: u32 = 0u; m < TILE; m = m + 1u) {
      // sum over m: aTile[m, kOut_in_tile=lid.y] · bTile[m, nOut_in_tile=lid.x]
      sum = sum + aTile[m * TILE + lid.y] * bTile[m * TILE + lid.x];
    }
    workgroupBarrier();
  }

  if (kOut < K && nOut < N) { C[kOut * N + nOut] = sum; }
}
