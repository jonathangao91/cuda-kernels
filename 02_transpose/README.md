# 02 — Transpose

**Date:** 5/8/2026 (planning) · **Update:** 5/9/2026 (naive impl)

## What I did

**Planning (5/8).** I intend to write naive and tiled implementations of matrix transpose.

The naive impl is straightforward: each thread is responsible for a single element in input matrix `A`. For `A[i, j]` (`i`, `j` from built-ins and 1D → 2D index mapping), write its value to `B[j, i]`. I expect this solution to exhibit suboptimal memory coalescing because one side of the access pattern will become strided for sufficiently large matrices.

The tiled impl will follow conventional tiling principles: define tile dims, stage per-block tiled reads from DRAM to shared memory, perform the same per-thread index conversions and swapping, then write back. AI should not materially change because the underlying algorithm is unchanged. The goal is memory subsystem efficiency, i.e. improved global-memory coalescing through better access patterns.

**Naive transpose (5/9).** Added a naive transpose implementation. Reimplemented error handling, event telemetry, and performance analysis by hand as practice and to gauge how much learning carried over from the vector-add kernel. Established benchmarks and proper execution configuration (2D grid / block aligned with row–col indexing).

## Performance

**Naive transpose** (kernel timed with CUDA events; bytes moved modeled as `2 × N² × sizeof(int)` read + write per element; peak from `cudaDeviceGetAttribute` DRAM formula).

| `N` | Achieved (GB/s) | Peak (GB/s) | % of peak | Headroom |
|-----|-----------------|-------------|-----------|----------|
| 256 | 0.264745 | 1008.096 | 0.026262% | 99.973738% |
| 1024 | 133.406612 | 1008.096 | 13.233523% | 86.766477% |
| 4096 | 225.657925 | 1008.096 | 22.384567% | 77.615433% |

## Next steps

- Implement **tiled** transpose and benchmark **naive vs tiled** in the same harness.
- Think about reducing boilerplate where it makes sense (e.g. sharing the `CHECK_ERROR` macro with vector add); incremental payoff may be smaller now that the harness pattern is familiar.

I expect the following to trip me up on the tiled path:

1. Shared memory syntax (first time).
2. Bank conflicts if I am not careful — remember **padding** (e.g. `32 × (32 + 1)` tile storage) so column striding in shared memory does not map every lane to the same bank; `+1` is the usual choice with `32 × 32` tiles on 32-bank hardware (`gcd(33, 32) = 1`).
