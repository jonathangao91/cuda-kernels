# cuda-kernels

CUDA kernels written from scratch as part of an ongoing transition from backend systems engineering toward ML systems infrastructure. The goal is hands-on understanding of GPU programming — memory hierarchies, occupancy, coalescing, shared memory tiling, and where bandwidth and latency actually go on a real device.

Each kernel lives in its own folder with the kernel itself, a host-side test/benchmark harness, and a short writeup of what I tried, what I measured, and what I learned. The writeups are at least as important as the code — they're how I make sure I'm building intuition rather than just typing CUDA syntax.

## Background

I'm working through *Programming Massively Parallel Processors* (Hwu, Kirk, Hajj, 4th ed.) alongside this repo. Kernels here roughly track the chapters I'm reading: simple element-wise ops first, then memory-hierarchy-aware patterns (tiling, shared memory), then more interesting ML primitives.

For the foundational ML / transformer work that preceded this, see [zero-to-hero-ai](https://github.com/jonathangao91/zero-to-hero-ai).

## Kernels

1. Vector addition (done)
2. Tranpose (done)

## Setup

Tested on an RTX 4090 using CUDA 13.2. Each kernel folder has its own build instructions; most are a single `nvcc` invocation.

## Status

Active.

Companion blog: jonathangao.bearblog.dev
