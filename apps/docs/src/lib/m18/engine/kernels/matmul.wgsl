// Generic tiled matmul. C = A · B   where A:[M,K], B:[K,N], C:[M,N], all row-major.
// Backs qkvMatmul, attnOutMatmul, ffnMatmul1, ffnMatmul2 in the forward pass.
// 16×16 tile; one workgroup writes one tile of C.

const TILE: u32 = 16u;

@group(0) @binding(0) var<storage, read>          A:    array<f32>;
@group(0) @binding(1) var<storage, read>          B:    array<f32>;
@group(0) @binding(2) var<storage, read_write>    C:    array<f32>;
@group(0) @binding(3) var<uniform>                dims: vec4<u32>; // (M, K, N, _)

var<workgroup> aTile: array<f32, 256>; // TILE*TILE
var<workgroup> bTile: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(
  @builtin(workgroup_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let M = dims.x; let K = dims.y; let N = dims.z;
  let row = gid.y * TILE + lid.y;
  let col = gid.x * TILE + lid.x;

  var sum: f32 = 0.0;
  let nTiles = (K + TILE - 1u) / TILE;

  for (var t: u32 = 0u; t < nTiles; t = t + 1u) {
    let aCol = t * TILE + lid.x;
    let bRow = t * TILE + lid.y;
    let aIdx = lid.y * TILE + lid.x;
    if (row < M && aCol < K) { aTile[aIdx] = A[row * K + aCol]; } else { aTile[aIdx] = 0.0; }
    if (bRow < K && col < N) { bTile[aIdx] = B[bRow * N + col]; } else { bTile[aIdx] = 0.0; }
    workgroupBarrier();

    for (var k: u32 = 0u; k < TILE; k = k + 1u) {
      sum = sum + aTile[lid.y * TILE + k] * bTile[k * TILE + lid.x];
    }
    workgroupBarrier();
  }

  if (row < M && col < N) { C[row * N + col] = sum; }
}
