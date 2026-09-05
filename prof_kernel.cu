#include "utils.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < M && y < N) {
    float tmp = 0.0f;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}

int main() {
  int M = 4096, N = 4096, K = 4096;
  float *dA, *dB, *dC;
  cudaMalloc(&dA, sizeof(float) * M * K);
  cudaMalloc(&dB, sizeof(float) * K * N);
  cudaMalloc(&dC, sizeof(float) * M * N);

  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32, 32);
  sgemm_naive<<<gridDim, blockDim>>>(M, N, K, 1.0f, dA, dB, 0.0f, dC);
  cudaDeviceSynchronize();

  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  return 0;
}
