
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

int main() {
  const int M = 4096, N = 4096, K = 4096, repeats = 5;
  const size_t bytes_a = size_t(M) * K * sizeof(float);
  const size_t bytes_b = size_t(K) * N * sizeof(float);
  const size_t bytes_c = size_t(M) * N * sizeof(float);
  float *A = nullptr, *B = nullptr, *C = nullptr;
  cublasHandle_t handle = nullptr;
  cudaEvent_t start = nullptr, stop = nullptr;
  cudaError_t cuda_status = cudaMalloc(&A, bytes_a);
  if (cuda_status != cudaSuccess) return 1;
  cuda_status = cudaMalloc(&B, bytes_b);
  if (cuda_status != cudaSuccess) return 1;
  cuda_status = cudaMalloc(&C, bytes_c);
  if (cuda_status != cudaSuccess) return 1;
  cudaMemset(A, 0, bytes_a); cudaMemset(B, 0, bytes_b); cudaMemset(C, 0, bytes_c);
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return 1;
  const float alpha = 1.0f, beta = 0.0f;
  if (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                  &alpha, A, M, B, K, &beta, C, M) != CUBLAS_STATUS_SUCCESS) return 1;
  cudaDeviceSynchronize();
  cudaEventCreate(&start); cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < repeats; ++i)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                &alpha, A, M, B, K, &beta, C, M);
  cudaEventRecord(stop); cudaEventSynchronize(stop);
  float milliseconds = 0.0f;
  cudaEventElapsedTime(&milliseconds, start, stop);
  const double seconds = milliseconds / 1000.0 / repeats;
  const double gflops = 2.0 * M * N * K / (seconds * 1.0e9);
  std::printf("cuBLAS,%.9f,%.3f\\n", seconds, gflops);
  cublasDestroy(handle); cudaEventDestroy(start); cudaEventDestroy(stop);
  cudaFree(A); cudaFree(B); cudaFree(C);
  return 0;
}
