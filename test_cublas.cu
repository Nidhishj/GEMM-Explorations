#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <sys/time.h>

inline void randomize_matrix(float *mat, int N) {
  struct timeval time {};
  gettimeofday(&time, nullptr);
  srand(time.tv_usec);
  for (int i = 0; i < N; i++) {
    mat[i] = (float)(rand() % 5) / 5.0f;
  }
}

int main() {
  int M = 4096, N = 4096, K = 4096;
  float alpha = 1.0f, beta = 0.0f;
  float *A = (float *)malloc(sizeof(float) * M * K);
  float *B = (float *)malloc(sizeof(float) * K * N);
  float *C = (float *)malloc(sizeof(float) * M * N);
  randomize_matrix(A, M * K); randomize_matrix(B, K * N); randomize_matrix(C, M * N);

  float *dA, *dB, *dC;
  cudaMalloc(&dA, sizeof(float) * M * K);
  cudaMalloc(&dB, sizeof(float) * K * N);
  cudaMalloc(&dC, sizeof(float) * M * N);
  cudaMemcpy(dA, A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, sizeof(float) * K * N, cudaMemcpyHostToDevice);

  cublasHandle_t handle;
  cublasCreate(&handle);

  // warmup
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
  cudaDeviceSynchronize();
  
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  int repeat = 5;
  cudaEventRecord(beg);
  for (int j = 0; j < repeat; j++) {
      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
  }
  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float elapsed_ms;
  cudaEventElapsedTime(&elapsed_ms, beg, end);
  double elapsed_seconds = elapsed_ms / 1000.0;
  double flops = 2.0 * M * N * K;
  double gflops = (repeat * flops * 1e-9) / elapsed_seconds;

  printf("cuBLAS GFLOPS: %.1f\n", gflops);
  printf("Elapsed time: %.6f seconds\n", elapsed_seconds);

  return 0;
}
