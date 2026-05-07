#include <cstdio>
#include <cuda_runtime.h>

// Takes two inputs a and b and sums them.
__global__ void vectorAdd(int *a, int *b, int *c, int len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) {
    c[idx] = a[idx] + b[idx];
  }
}

int main() {
  const size_t len = 10;
  int a[len];
  int b[len];
  int c[len];

  for (size_t i = 0; i < len; i++) {
    a[i] = i;
    b[i] = len - 1 - i;
  }

  int *a_d;
  int *b_d;
  int *c_d;

  cudaMalloc(&a_d, sizeof(int) * len);
  cudaMalloc(&b_d, sizeof(int) * len);
  cudaMalloc(&c_d, sizeof(int) * len);

  cudaMemcpy(a_d, a, sizeof(int) * len, cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b, sizeof(int) * len, cudaMemcpyHostToDevice);

  int threads_per_block = 32;
  int blocks = (len + threads_per_block - 1) / threads_per_block;
  vectorAdd<<<blocks, threads_per_block>>>(a_d, b_d, c_d, len);

  cudaMemcpy(c, c_d, sizeof(int) * len, cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < len; i++) {
    printf("Idx %zu in vector add output equals to %i.\n", i, c[i]);
  }

  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);
}
