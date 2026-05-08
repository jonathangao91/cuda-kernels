# 02 — Transpose

**Date:** 5/8/2026

## What I did

Planning. I intend to write naive and tiled implementations of matrix transpose.

The naive impl is straightforward: each thread is responsible for a single element in input matrix `A`. For `A[i, j]` (`i, j` identified via built-ins and 1D -> 2D index conversion), write its value to `B[j, i]`. I expect this solution to exhibit suboptimal memory coalescing because one side of the access pattern will become strided for sufficiently large matrices.

The tiled impl will follow conventional tiling principles: define tile dims, stage per-block tiled reads from DRAM to shared memory, perform the same per-thread index conversions and swapping, then write back. AI should not materially change because the underlying algorithm is unchanged. The goal is memory subsystem efficiency, i.e. improved global-memory coalescing through better access patterns.

## Next steps

Implement transpose and benchmark naive vs tiled.

I also want to think more about reducing boilerplate where possible. My learning might end up being that I can refactor the error-handling macro from vector add for reuse and little else. While writing the full harness in the previous kernel was a great learning exercise, my incremental learning from rewriting all of that again here will likely be lower.