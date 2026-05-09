#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_ERROR(res)                                                       \
  do {                                                                         \
    cudaError_t err = res;                                                     \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA error at line %i in file %s: %s\n", __LINE__, __FILE__,     \
             cudaGetErrorString(err));                                         \
      rc = err;                                                                \
      goto cleanup;                                                            \
    }                                                                          \
  } while (0);

__global__ void naive_transpose(int *a, int *at, int dim) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < dim && col < dim) {
    size_t src_1d_idx = static_cast<size_t>(row) * dim + col;
    size_t dst_1d_idx = static_cast<size_t>(col) * dim + row;

    at[dst_1d_idx] = a[src_1d_idx];
  }

  return;
}

int main() {
  int rc = cudaSuccess;
  const int dim_benchmarks[3] = {256, 1024, 4096};

  for (int N : dim_benchmarks) {
    int mismatch_count = 0;
    std::vector<int> a(N * N);
    std::vector<int> at(N * N);

    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        a[i * N + j] = i + j;
      }
    }

    int *a_d = nullptr;
    int *at_d = nullptr;

    // Exec config
    const int tile_dim = 32;
    const dim3 block(tile_dim, tile_dim);
    const dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    cudaError_t exec_err = cudaSuccess;

    // Telemetry
    cudaEvent_t start = nullptr;
    cudaEvent_t end = nullptr;
    float elapsed_ms = 0;
    const double bytes_moved = static_cast<size_t>(N) * N * sizeof(int) * 2;
    double actual_bw = 0;
    int dev = 0;
    int bus_width_bits = 0;
    int clock_rate_khz = 0;
    double peak_bw = 0.0;
    double pct_of_peak = 0.0;
    double headroom_pct = 0.0;

    CHECK_ERROR(cudaGetDevice(&dev));
    CHECK_ERROR(cudaDeviceGetAttribute(&bus_width_bits,
                                       cudaDevAttrGlobalMemoryBusWidth, dev));
    CHECK_ERROR(cudaDeviceGetAttribute(&clock_rate_khz,
                                       cudaDevAttrMemoryClockRate, dev));

    // DDR (2) * 1000 * clock_rate_khz * (bus_width_bits / 8.0) / 1e9 -> GB/s
    peak_bw = (2.0 * 1000.0 * clock_rate_khz * (bus_width_bits / 8.0)) / 1e9;

    CHECK_ERROR(cudaMalloc(&a_d, static_cast<size_t>(N) * N * sizeof(int)));
    CHECK_ERROR(cudaMalloc(&at_d, static_cast<size_t>(N) * N * sizeof(int)));

    CHECK_ERROR(cudaMemcpy(a_d, a.data(), a.size() * sizeof(int),
                           cudaMemcpyHostToDevice));

    CHECK_ERROR(cudaEventCreate(&start));
    CHECK_ERROR(cudaEventCreate(&end));

    CHECK_ERROR(cudaEventRecord(start));
    naive_transpose<<<grid, block>>>(a_d, at_d, N);
    // Catches kernel launch errors.
    if ((exec_err = cudaGetLastError()) != cudaSuccess) {
      printf("Kernel launch failed: %s\n", cudaGetErrorString(exec_err));
      rc = exec_err;
      goto cleanup;
    }

    CHECK_ERROR(cudaEventRecord(end));
    // Catches kernel runtime errors.
    CHECK_ERROR(cudaEventSynchronize(end));

    CHECK_ERROR(cudaMemcpy(at.data(), at_d, at.size() * sizeof(int),
                           cudaMemcpyDeviceToHost));

    // Validate
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        size_t at_1d_idx = static_cast<size_t>(i) * N + j;
        size_t a_1d_idx = static_cast<size_t>(j) * N + i;
        if (at[at_1d_idx] != a[a_1d_idx]) {
          printf(
              "at[%zu] containing %i which corresponds to A_t[%i,%i] does not "
              "equal a[%zu] containing %i, which corresponds to A[%i,%i].\n",
              at_1d_idx, at[at_1d_idx], i, j, a_1d_idx, a[a_1d_idx], j, i);
          mismatch_count++;
        }
      }
    }

    // Telemetry
    CHECK_ERROR(cudaEventElapsedTime(&elapsed_ms, start, end));
    // Likely mem-bound, so compute bandwidth.
    actual_bw = (bytes_moved / 1e9) / (elapsed_ms / 1000.0);
    printf("Actual bandwidth for dim %i: %f GB/s\n", N, actual_bw);
    printf("Peak bandwidth: %f GB/s\n", peak_bw);
    pct_of_peak = (actual_bw / peak_bw) * 100.0;
    headroom_pct = 100.0 - pct_of_peak;
    printf("%f%% of peak with %f%% headroom for dim %i.\n", pct_of_peak,
           headroom_pct, N);

  cleanup:

    cudaError_t free_err = cudaSuccess;
    if (a_d && (free_err = cudaFree(a_d)) != cudaSuccess) {
      printf("Failed to free a_d: %s\n", cudaGetErrorString(free_err));
      rc = free_err;
    }
    if (at_d && (free_err = cudaFree(at_d)) != cudaSuccess) {
      printf("Failed to free at_d: %s\n", cudaGetErrorString(free_err));
      rc = free_err;
    }

    cudaError_t destroy_err = cudaSuccess;
    if (start && (destroy_err = cudaEventDestroy(start)) != cudaSuccess) {
      printf("Failed to destroy start event: %s\n",
             cudaGetErrorString(destroy_err));
      rc = destroy_err;
    }
    if (end && (destroy_err = cudaEventDestroy(end)) != cudaSuccess) {
      printf("Failed to destroy end event: %s\n",
             cudaGetErrorString(destroy_err));
      rc = destroy_err;
    }

    if (rc != cudaSuccess) {
      return rc;
    } else if (mismatch_count > 0) {
      return 1;
    }
  }

  return 0;
}
