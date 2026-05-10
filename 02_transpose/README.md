# 02 — Transpose

**Date:** 5/8/2026 (planning) · **Update:** 5/9/2026 (naive + tiled impl)

## What I did

**Planning (5/8).** I intend to write naive and tiled implementations of matrix transpose.

The naive impl is straightforward: each thread is responsible for a single element in input matrix `A`. For `A[i, j]` (`i`, `j` from built-ins and 1D → 2D index mapping), write its value to `B[j, i]`. I expect this solution to exhibit suboptimal memory coalescing because one side of the access pattern will become strided for sufficiently large matrices.

The tiled impl will follow conventional tiling principles: define tile dims, stage per-block tiled reads from DRAM to shared memory, perform the same per-thread index conversions and swapping, then write back. **Arithmetic intensity** (*AI*) should not materially change because the underlying algorithm is unchanged; only the memory access pattern does. The goal is memory subsystem efficiency, i.e. improved global-memory coalescing through better access patterns.

**Naive transpose (5/9).** Added a naive transpose implementation. Reimplemented error handling, event telemetry, and performance analysis by hand as practice and to gauge how much learning carried over from the vector-add kernel. Established benchmarks and proper execution configuration (2D grid / block aligned with row–col indexing).

## Performance

Kernels timed with CUDA events; bytes moved modeled as `2 × N² × sizeof(int)` (read + write per element); peak from `cudaDeviceGetAttribute` DRAM formula.

### Naive transpose

| `N` | Achieved (GB/s) | Peak (GB/s) | % of peak | Headroom |
|-----|-----------------|-------------|-----------|----------|
| 256 | 0.264745 | 1008.096 | 0.026262% | 99.973738% |
| 1024 | 133.406612 | 1008.096 | 13.233523% | 86.766477% |
| 4096 | 225.657925 | 1008.096 | 22.384567% | 77.615433% |

### Tiled transpose

| `N` | Achieved (GB/s) | Peak (GB/s) | % of peak | Headroom |
|-----|-----------------|-------------|-----------|----------|
| 256 | 1.586674 | 1008.096 | 0.157393% | 99.842607% |
| 1024 | 264.791927 | 1008.096 | 26.266539% | 73.733461% |
| 4096 | 591.663688 | 1008.096 | 58.691205% | 41.308795% |

## Next steps

- Think about reducing boilerplate where it makes sense (e.g. sharing the `CHECK_ERROR` macro with vector add); incremental payoff may be smaller now that the harness pattern is familiar.

## Notes — tiled path (5/9)

I expect the following to trip me up on the tiled path:

1. Shared memory syntax (first time).
2. Bank conflicts if I am not careful — remember **padding** (e.g. `32 × (32 + 1)` tile storage) so column striding in shared memory does not map every lane to the same bank; `+1` is the usual choice with `32 × 32` tiles on 32-bank hardware (`gcd(33, 32) = 1`).

Implemented tiled transpose with a large memory-throughput improvement vs naive at larger `N`. The harness launches **tiled** transpose only (no dual-kernel mash in one binary).

What surprised me:

1. Without understanding the purpose of tiling, it is easy to mess up what threads should read from and write to which elements in global memory. While fighting the implementation to pass validation, I produced a **correct** phase 2 that still **failed to coalesce** global stores.
2. I introduced a subtle bug: swapping **both** global addresses and tile indices in phase 2 effectively **undid** the transpose. Two coherent fixes: **(a)** flip only the tile access (classic NVIDIA pattern; coalesced writes), or **(b)** read the tile the same way it was filled and keep swapped global indices (strided writes; no coalescing win).
3. I did not reason about blocks carefully enough at first and used nested loops over all tiles per thread, which defeats parallelism across the grid. **Each block handles one tile**, not the whole matrix serially.
