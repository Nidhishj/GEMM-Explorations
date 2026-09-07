#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <sys/time.h>

inline void randomize_matrix(float *mat, int N) {
  struct timeval time {};
  gettimeofday(&time, nullptr);
  srand(time.tv_usec);
  for (int i = 0; i < N; i++) {
    float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
    tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
    mat[i] = tmp;
  }
}

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}

void run_kernel(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
  #define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32, 32);
  sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
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
  cudaMemcpy(dC, C, sizeof(float) * M * N, cudaMemcpyHostToDevice);

  run_kernel(M, N, K, alpha, dA, dB, beta, dC);
  cudaDeviceSynchronize();
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("CUDA Error: %s\n", cudaGetErrorString(err));
  } else {
      printf("Kernel executed successfully!\n");
  }
  return 0;
}
