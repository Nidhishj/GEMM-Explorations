#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  std::fprintf(stderr, "CUDA error: %s\\n", cudaGetErrorString(e)); std::exit(1); \
} } while (0)
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

template<int BM, int BN, int BK, int TM, int TN>
__global__ void tiled_gemm(int M, int N, int K, const float* A,
                           const float* B, float* C) {
  constexpr int THREADS = (BM / TM) * (BN / TN);
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];
  const int tid = threadIdx.x;
  const int thread_row = tid / (BN / TN);
  const int thread_col = tid % (BN / TN);
  float accum[TM * TN] = {0.0f};

  for (int k0 = 0; k0 < K; k0 += BK) {
    for (int i = tid; i < BM * BK; i += THREADS) {
      const int r = i / BK, c = i % BK;
      const int gr = blockIdx.y * BM + r, gc = k0 + c;
      As[i] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
    }
    for (int i = tid; i < BK * BN; i += THREADS) {
      const int r = i / BN, c = i % BN;
      const int gr = k0 + r, gc = blockIdx.x * BN + c;
      Bs[i] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
    }
    __syncthreads();
    for (int k = 0; k < BK; ++k) {
      for (int r = 0; r < TM; ++r) {
        const float a = As[(thread_row * TM + r) * BK + k];
        for (int c = 0; c < TN; ++c)
          accum[r * TN + c] += a * Bs[k * BN + thread_col * TN + c];
      }
    }
    __syncthreads();
  }
  for (int r = 0; r < TM; ++r) for (int c = 0; c < TN; ++c) {
    const int gr = blockIdx.y * BM + thread_row * TM + r;
    const int gc = blockIdx.x * BN + thread_col * TN + c;
    if (gr < M && gc < N) C[gr * N + gc] = accum[r * TN + c];
  }
}

struct Config { int bm, bn, bk, tm, tn; };

bool legal(Config c) {
  const int threads = (c.bm / c.tm) * (c.bn / c.tn);
  const int smem = (c.bm * c.bk + c.bk * c.bn) * 4;
  return c.bm % c.tm == 0 && c.bn % c.tn == 0 && threads >= 32 &&
         threads <= 1024 && threads % 32 == 0 && smem <= 48 * 1024;
}

template<int BM, int BN, int BK, int TM, int TN>
float measure(int M, int N, int K, const float* A, const float* B,
              float* C, int repeats) {
  constexpr int threads = (BM / TM) * (BN / TN);
  dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  tiled_gemm<BM, BN, BK, TM, TN><<<grid, threads>>>(M, N, K, A, B, C);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < repeats; ++i)
    tiled_gemm<BM, BN, BK, TM, TN><<<grid, threads>>>(M, N, K, A, B, C);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / repeats;
}

template<int BM, int BN, int BK, int TM, int TN>
void record_result(const Config& c, int M, int N, int K, const float* dA,
                  const float* dB, float* dC, int repeats, std::ofstream& csv) {
  const float ms = measure<BM, BN, BK, TM, TN>(M, N, K, dA, dB, dC, repeats);
  const double gflops = 2.0 * M * N * K / (ms * 1.0e6);
  const int threads = (BM / TM) * (BN / TN);
  csv << M << ',' << N << ',' << K << ',' << c.bm << ',' << c.bn << ','
      << c.bk << ',' << c.tm << ',' << c.tn << ',' << threads << ','
      << ms << ',' << gflops << std::endl;
  std::cout << "BM=" << c.bm << " BN=" << c.bn << " BK=" << c.bk
            << " TM=" << c.tm << " TN=" << c.tn << " -> "
            << gflops << " GFLOP/s" << std::endl;
}

int main(int argc, char** argv) {
  const int M = argc > 1 ? std::atoi(argv[1]) : 2048;
  const int N = argc > 2 ? std::atoi(argv[2]) : 2048;
  const int K = argc > 3 ? std::atoi(argv[3]) : 2048;
  const int repeats = argc > 4 ? std::atoi(argv[4]) : 10;
  const size_t bytes_a = size_t(M) * K * sizeof(float);
  const size_t bytes_b = size_t(K) * N * sizeof(float);
  const size_t bytes_c = size_t(M) * N * sizeof(float);
  std::vector<float> hA(size_t(M) * K, 0.01f), hB(size_t(K) * N, 0.02f);
  float *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc(&dA, bytes_a));
  CUDA_CHECK(cudaMalloc(&dB, bytes_b));
  CUDA_CHECK(cudaMalloc(&dC, bytes_c));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes_a, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes_b, cudaMemcpyHostToDevice));
  std::ofstream csv("parameter_sweep.csv");
  csv << "M,N,K,BM,BN,BK,TM,TN,Threads,ElapsedMs,GFLOPS" << std::endl;

  // Exclude 1024-thread/high-register configurations that fail on this GPU.
  const std::vector<Config> configs = {
      {64,64,8,4,4}, {64,64,8,8,8}, {64,64,16,8,8},
      {128,64,8,8,4}, {128,64,16,8,4},
      {128,128,8,8,8}, {128,128,16,8,8}
  };
  for (const Config& c : configs) {
    if (!legal(c)) continue;
    CUDA_CHECK(cudaMemset(dC, 0, bytes_c));
    if (c.bm == 64 && c.bk == 8 && c.tm == 4)
      record_result<64,64,8,4,4>(c, M, N, K, dA, dB, dC, repeats, csv);
    else if (c.bm == 64 && c.bk == 8)
      record_result<64,64,8,8,8>(c, M, N, K, dA, dB, dC, repeats, csv);
    else if (c.bm == 64)
      record_result<64,64,16,8,8>(c, M, N, K, dA, dB, dC, repeats, csv);
    else if (c.bn == 64 && c.bk == 8)
      record_result<128,64,8,8,4>(c, M, N, K, dA, dB, dC, repeats, csv);
    else if (c.bn == 64)
      record_result<128,64,16,8,4>(c, M, N, K, dA, dB, dC, repeats, csv);
    else if (c.bk == 8)
      record_result<128,128,8,8,8>(c, M, N, K, dA, dB, dC, repeats, csv);
    else
      record_result<128,128,16,8,8>(c, M, N, K, dA, dB, dC, repeats, csv);
  }
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
}
