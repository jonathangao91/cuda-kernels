#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_ERROR(res)                                                       \
  do {                                                                         \
    cudaError_t err = res;                                                     \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA error: %s at line %i in file %s\n",                         \
             cudaGetErrorString(err), __LINE__, __FILE__);                     \
      rc = err;                                                                \
      goto cleanup;                                                            \
    }                                                                          \
  } while (0);

// Takes two inputs a and b and sums them.
__global__ void vectorAdd(int *a, int *b, int *c, int len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) {
    c[idx] = a[idx] + b[idx];
  }
}

int main() {

  const size_t benchmarks[3] = {static_cast<size_t>(1e6),
                                static_cast<size_t>(1e7),
                                static_cast<size_t>(1e8)};
  for (size_t benchmark : benchmarks) {
    int threads_per_block = 32;
    int blocks = (benchmark + threads_per_block - 1) / threads_per_block;

    std::vector<int> a(benchmark), b(benchmark), c(benchmark);

    for (size_t i = 0; i < benchmark; i++) {
      a[i] = i;
      b[i] = benchmark - 1 - i;
    }

    int *a_d = nullptr;
    int *b_d = nullptr;
    int *c_d = nullptr;

    int rc = 0;

    // Catch launch errors.
    cudaError_t exec_err = cudaSuccess;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    float ms = 0.0f;
    // Bytes r/w: 3 vectors, same length.
    // a and b are input vectors read in their entirety.
    // c is the output vector, written in its entirety.
    // Therefore, memory read in bytes is 2 * len * sizeof(int).
    // Memory written is len * sizeof(int).
    // Total memory moved is 3 * len * sizeof(int).
    size_t memory_moved = 3 * benchmark * sizeof(int);
    double bandwidth = 0.0;
    int mem_clock_khz = 0;
    int bus_width_bits = 0;
    int dev = 0;
    double peak_bandwidth = 0.0;
    double pct_of_peak = 0.0;
    double headroom_pct = 0.0;

    CHECK_ERROR(cudaGetDevice(&dev));
    CHECK_ERROR(cudaDeviceGetAttribute(&mem_clock_khz,
                                       cudaDevAttrMemoryClockRate, dev));
    CHECK_ERROR(cudaDeviceGetAttribute(&bus_width_bits,
                                       cudaDevAttrGlobalMemoryBusWidth, dev));
    peak_bandwidth =
        (2.0 * bus_width_bits / 8.0) * (mem_clock_khz * 1000.0) / 1e9;

    CHECK_ERROR(cudaMalloc(&a_d, sizeof(int) * benchmark));
    CHECK_ERROR(cudaMalloc(&b_d, sizeof(int) * benchmark));
    CHECK_ERROR(cudaMalloc(&c_d, sizeof(int) * benchmark));

    CHECK_ERROR(cudaMemcpy(a_d, a.data(), sizeof(int) * benchmark,
                           cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(b_d, b.data(), sizeof(int) * benchmark,
                           cudaMemcpyHostToDevice));

    CHECK_ERROR(cudaEventCreate(&start));
    CHECK_ERROR(cudaEventCreate(&stop));
    CHECK_ERROR(cudaEventRecord(start));

    vectorAdd<<<blocks, threads_per_block>>>(a_d, b_d, c_d, benchmark);
    exec_err = cudaGetLastError();

    CHECK_ERROR(cudaEventRecord(stop));
    CHECK_ERROR(cudaEventSynchronize(stop));
    CHECK_ERROR(cudaEventElapsedTime(&ms, start, stop));

    if (exec_err != cudaSuccess) {
      printf("Kernel launch failed at line %i in file %s: %s\n", __LINE__,
             __FILE__, cudaGetErrorString(exec_err));
      rc = exec_err;
      goto cleanup;
    }

    // Catch execution-time errors.
    CHECK_ERROR(cudaDeviceSynchronize());

    CHECK_ERROR(cudaMemcpy(c.data(), c_d, sizeof(int) * benchmark,
                           cudaMemcpyDeviceToHost));

    // Validate outcome.
    for (size_t i = 0; i < benchmark; i++) {
      if (c[i] != a[i] + b[i]) {
        printf("Unexpected kernel output for benchmark of vector size %zu: "
               "c[%zu] of %i does not equal to a[%zu] of "
               "%i + b[%zu] of %i.\n",
               benchmark, i, c[i], i, a[i], i, b[i]);
      }
    }

    // Performance
    bandwidth = (memory_moved / 1e9) / (ms / 1000.0);
    printf("Bandwidth of %f measured for benchmark of vector size %zu.\n",
           bandwidth, benchmark);
    pct_of_peak = (bandwidth / peak_bandwidth) * 100.0;
    headroom_pct = 100.0 - pct_of_peak;
    printf("Operating at %f%% of peak bandwidth. %f%% headroom.\n", pct_of_peak,
           headroom_pct);

  cleanup:

    cudaError_t destroy_err = cudaSuccess;
    if (start && (destroy_err = cudaEventDestroy(start)) != cudaSuccess) {
      printf("Failed to destroy start event: %s\n",
             cudaGetErrorString(destroy_err));
      rc = destroy_err;
    }
    if (stop && (destroy_err = cudaEventDestroy(stop)) != cudaSuccess) {
      printf("Failed to destroy stop event: %s\n",
             cudaGetErrorString(destroy_err));
      rc = destroy_err;
    }

    cudaError_t free_err = cudaSuccess;
    if (a_d && (free_err = cudaFree(a_d)) != cudaSuccess) {
      printf("CUDA error on free at line %i in file %s: %s\n", __LINE__,
             __FILE__, cudaGetErrorString(free_err));
      rc = free_err;
    }
    if (b_d && (free_err = cudaFree(b_d)) != cudaSuccess) {
      printf("CUDA error on free at line %i in file %s: %s\n", __LINE__,
             __FILE__, cudaGetErrorString(free_err));
      rc = free_err;
    }
    if (c_d && (free_err = cudaFree(c_d)) != cudaSuccess) {
      printf("CUDA error on free at line %i in file %s: %s\n", __LINE__,
             __FILE__, cudaGetErrorString(free_err));
      rc = free_err;
    }

    if (rc != cudaSuccess) {
      return rc;
    }
  }

  return cudaSuccess;
}
