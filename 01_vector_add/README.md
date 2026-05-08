# 01 — Vector Add

**Date:** 5/7/2026  
**Hardware:** RTX 4090  
**CUDA:** V13.2  
**OS:** WSL2 on Windows  

## What I did

I have been reading PMPP on and off since February.  I circled back to an exercise in chapter 2 and coded up a vector add kernel.  My dev cycle was to jot down whatever I remembered, then relying on syntax / semantics diagnostics, and finally asking an LLM to assess my work and provide minimal (directional-only) pointers for yellow and red flags.

I set up the input arrays such that each output element would sum up to the same value for simple validation.

```bash
nvcc -arch=sm_89 vector_add.cu -o vector_add
./vector_add
```

## Performance

As reported by the harness:

- `N=1,000,000`: `4.843 GB/s` (`0.480%` of peak, `99.520%` headroom)
- `N=10,000,000`: `459.784 GB/s` (`45.609%` of peak, `54.391%` headroom)
- `N=100,000,000`: `735.366 GB/s` (`72.946%` of peak, `27.054%` headroom)

Interpretation:

- The smallest benchmark is dominated by fixed overhead (launch + setup), so achieved bandwidth is low.
- As `N` grows, throughput climbs and gets much closer to the memory subsystem limit.
- For this memory-bound kernel, this is expected and gives me a clean baseline before moving to transpose.

## What surprised me

1. I realized how easy it is to mix up host and device vars without adhering to proper naming conventions.
2. I was surprised by how terribly learnings committed to memory from passively reading PMPP maps to actual code.  The intuition was there, but the syntax absolutely was not.  Once I was reminded of a few basic concepts (exactly which dunder annotations are important and when, built-ins and how to access them, memory allocation / copying basics), a first working draft of vector-add was born.
3. My first attempt at verifying correctness was simply `nvcc vector_add.cu`.  The output contained only 0s.  Running again using `-arch=sm_89` did the trick, as suggested by an LLM.  My understanding is that the GPU exposed to my CUDA on WSL setup (RTX 4090) uses 89 architecture (Ada), so whatever default architecture was used to compile my code must've been fundamentally incompatible with my GPU.  However, the bigger point is that CUDA fails silently.

## 5/8/2026 Update

### What I did

Added error handling.  To this end, introduced a macro that wraps all CUDA API calls, except for calls in the cleanup block since cleanup should not defer to itself.

Added benchmark vector sizes of `1M`, `10M`, and `100M`.  Since vector add has low arithmetic intensity (for every ~3 memory ops, ~1 FLOP), it is memory bound.  This means the metric I should focus on is memory throughput (`GB/s`).  I benchmarked actual memory throughput and compared measurements against theoretical RTX 4090 peak.  To this end, I added telemetry to record start/stop CUDA events and calculate elapsed time.

### What surprised me

1. How incredibly verbose CUDA error handling is.  Everything can return an error and every error must be checked and reported immediately.  Otherwise, the program loses debuggabilty due to loss of error attribution when error reporting is not immediately preceded by the API response being reported.
2. In general, how incredibly verbose the kernel harness is.  I must've spent 95% of my time writing host code, not kernel code.
3. Numerous gotchas: cleanup goto statement infinite loops, resource management.  Aside from the verbosity of CUDA, this was a great learning experience in C++ development, which at Google was primarily about maintaining a legacy system, where incremental code changes were typically copy-paste with minor tweaks (e.g. tweaking some business logic followed by extending an existing test suite).
4. I finally learned what DDR means when I was learning about how to calculate peak BW.  I have come across this term while building PCs.

## Next steps

Implement transpose in two versions: naive transpose and tiled shared-memory transpose. Benchmark both and focus on the bandwidth gap driven by memory access pattern/coalescing.