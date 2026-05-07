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

TBD.  I did not get a chance to benchmark my implementation.

## What surprised me

1. I realized how easy it is to mix up host and device vars without adhering to proper naming conventions.
2. I was surprised by how terribly learnings committed to memory from passively reading PMPP maps to actual code.  The intuition was there, but the syntax absolutely was not.  Once I was reminded of a few basic concepts (exactly which dunder annotations are important and when, built-ins and how to access them, memory allocation / copying basics), a first working draft of vector-add was born.
3. My first attempt at verifying correctness was simply `nvcc vector_add.cu`.  The output contained only 0s.  Running again using `-arch=sm_89` did the trick, as suggested by an LLM.  My understanding is that the GPU exposed to my CUDA on WSL setup (RTX 4090) uses 89 architecture (Ada), so whatever default architecture was used to compile my code must've been fundamentally incompatible with my GPU.  However, the bigger point is that CUDA fails silently.

## Next steps

The plan is to add error handling next as part of building strong CUDA habits, then to benchmark the implementation on various input configurations, as telemetry plays a central role in performance engineering.