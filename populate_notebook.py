import nbformat as nbf
import os

nb_path = '/ssd_scratch/nidhishj/GEMM-Explorations/GEMM_Assignment.ipynb'
if not os.path.exists(nb_path):
    print(f"Notebook {nb_path} not found.")
    exit(1)

utils_path = os.path.join(os.path.dirname(nb_path), 'utils.cuh')

with open(nb_path, 'r') as f:
    nb = nbf.read(f, as_version=4)

# Find the cell where we should start adding new cells
# The original notebook has a Naive Matmul cell. We will just append to the end of the notebook or replace the placeholder cells.

# Let's create the utils.cuh cell
utils_code = """%%writefile __UTILS_PATH__
#pragma once
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
"""
utils_code = utils_code.replace('__UTILS_PATH__', utils_path)

cells_to_add = [
    nbf.v4.new_markdown_cell("### Utility Header\nRun this cell to write the `utils.cuh` header containing our matrix initialization routines."),
    nbf.v4.new_code_cell(utils_code)
]

kernels = [
    {
        "name": "1_naive",
        "title": "Kernel 1: Naive Matrix Multiplication",
        "file": "/ssd_scratch/nidhishj/SGEMM_CUDA_REF/src/kernels/1_naive.cuh",
        "launch": """
void run_kernel(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32, 32);
  sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
"""
    },
    {
        "name": "2_kernel_global_mem_coalesce",
        "title": "Kernel 2: Global Memory Coalescing",
        "file": "/ssd_scratch/nidhishj/SGEMM_CUDA_REF/src/kernels/2_kernel_global_mem_coalesce.cuh",
        "launch": """
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
void run_kernel(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32 * 32);
  sgemm_global_mem_coalesce<32><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
"""
    },
    {
        "name": "3_kernel_shared_mem_blocking",
        "title": "Kernel 3: Shared Memory Cache-Blocking",
        "file": "/ssd_scratch/nidhishj/SGEMM_CUDA_REF/src/kernels/3_kernel_shared_mem_blocking.cuh",
        "launch": """
void run_kernel(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32 * 32);
  cudaFuncSetAttribute(sgemm_shared_mem_block<32>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
  sgemm_shared_mem_block<32><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
"""
    },
    {
        "name": "4_kernel_1D_blocktiling",
        "title": "Kernel 4: 1D Blocktiling",
        "file": "/ssd_scratch/nidhishj/SGEMM_CUDA_REF/src/kernels/4_kernel_1D_blocktiling.cuh",
        "launch": """
void run_kernel(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
  const uint BM = 64;
  const uint BN = 64;
  const uint BK = 8;
  const uint TM = 8;
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim((BM * BN) / TM);
  sgemm1DBlocktiling<BM, BN, BK, TM><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
"""
    },
    {
        "name": "5_kernel_2D_blocktiling",
        "title": "Kernel 5: 2D Blocktiling",
        "file": "/ssd_scratch/nidhishj/SGEMM_CUDA_REF/src/kernels/5_kernel_2D_blocktiling.cuh",
        "launch": """
void run_kernel(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  const uint BM = 128;
  const uint BN = 128;
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim((BM * BN) / (TM * TN));
  sgemm2DBlocktiling<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
"""
    },
    {
        "name": "6_kernel_vectorize",
        "title": "Kernel 6: Vectorized SMEM and GMEM Accesses",
        "file": "/ssd_scratch/nidhishj/SGEMM_CUDA_REF/src/kernels/6_kernel_vectorize.cuh",
        "launch": """
void run_kernel(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  const uint BM = 128;
  const uint BN = 128;
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim((BM * BN) / (TM * TN));
  sgemmVectorize<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
"""
    },
    {
        "name": "9_kernel_autotuned",
        "title": "Kernel 9: Autotuning",
        "file": "/ssd_scratch/nidhishj/SGEMM_CUDA_REF/src/kernels/9_kernel_autotuned.cuh",
        "launch": """
void run_kernel(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
  const uint K9_BK = 16;
  const uint K9_TM = 8;
  const uint K9_TN = 8;
  const uint K9_BM = 128;
  const uint K9_BN = 128;
  dim3 blockDim(K9_NUM_THREADS);
  dim3 gridDim(CEIL_DIV(N, K9_BN), CEIL_DIV(M, K9_BM));
  sgemmAutotuned<K9_BM, K9_BN, K9_BK, K9_TM, K9_TN><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
"""
    },
    {
        "name": "10_kernel_warptiling",
        "title": "Kernel 10: Warptiling",
        "file": "/ssd_scratch/nidhishj/SGEMM_CUDA_REF/src/kernels/10_kernel_warptiling.cuh",
        "launch": """
void run_kernel(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
  const uint K10_NUM_THREADS = 128;
  const uint K10_BN = 128;
  const uint K10_BM = 128;
  const uint K10_BK = 16;
  const uint K10_WM = 64;
  const uint K10_WN = 64;
  const uint K10_WNITER = 4;
  const uint K10_TN = 4;
  const uint K10_TM = 8;
  dim3 blockDim(K10_NUM_THREADS);
  dim3 gridDim(CEIL_DIV(N, K10_BN), CEIL_DIV(M, K10_BM));
  sgemmWarptiling<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM, K10_TN, K10_NUM_THREADS><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
"""
    }
]

main_template = """
int main() {
  int M = 4096;
  int N = 4096;
  int K = 4096;
  float alpha = 1.0f;
  float beta = 0.0f;

  float *A = (float *)malloc(sizeof(float) * M * K);
  float *B = (float *)malloc(sizeof(float) * K * N);
  float *C = (float *)malloc(sizeof(float) * M * N);

  randomize_matrix(A, M * K);
  randomize_matrix(B, K * N);
  randomize_matrix(C, M * N);

  float *dA, *dB, *dC;
  cudaMalloc(&dA, sizeof(float) * M * K);
  cudaMalloc(&dB, sizeof(float) * K * N);
  cudaMalloc(&dC, sizeof(float) * M * N);

  cudaMemcpy(dA, A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, sizeof(float) * K * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dC, C, sizeof(float) * M * N, cudaMemcpyHostToDevice);

  // Warmup
  run_kernel(M, N, K, alpha, dA, dB, beta, dC);
  cudaDeviceSynchronize();

  int repeat_times = 5;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  cudaEventRecord(beg);
  for (int j = 0; j < repeat_times; j++) {
    run_kernel(M, N, K, alpha, dA, dB, beta, dC);
  }
  cudaEventRecord(end);
  cudaEventSynchronize(beg);
  cudaEventSynchronize(end);
  
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, beg, end);
  elapsed_time /= 1000.0; // Convert to seconds

  long flops = 2L * M * N * K;
  float gflops = (repeat_times * flops * 1e-9) / elapsed_time;

  printf("kernel_%s: %.1f GFLOPS\\n", "{NAME}", gflops);

  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  free(A); free(B); free(C);
  
  return 0;
}
"""

for k in kernels:
    cells_to_add.append(nbf.v4.new_markdown_cell(f"### {k['title']}"))
    with open(k['file'], 'r') as f:
        code = f.read()
    
    # Strip #pragma once and includes to keep the cell clean if we want,
    # but the easiest way is just to append our stuff.
    full_code = f"%%cuda\n#include \"{utils_path}\"\n\n{code}\n\n{k['launch']}\n\n{main_template.replace('{NAME}', k['name'])}"
    cells_to_add.append(nbf.v4.new_code_cell(full_code))


# Add profiling example cell
prof_code = """%%cuda --name prof_kernel.cu --compile false
#include "__UTILS_PATH__"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
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

int main() {
  int M = 1024, N = 1024, K = 1024;
  float *dA, *dB, *dC;
  cudaMalloc(&dA, sizeof(float) * M * K);
  cudaMalloc(&dB, sizeof(float) * K * N);
  cudaMalloc(&dC, sizeof(float) * M * N);

  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32, 32);
  sgemm_naive<<<gridDim, blockDim>>>(M, N, K, 1.0, dA, dB, 0.0, dC);
  cudaDeviceSynchronize();
  
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  return 0;
}
"""
prof_code = prof_code.replace('__UTILS_PATH__', utils_path)

cells_to_add.append(nbf.v4.new_markdown_cell("### Nsight Compute Profiling\nThe cell below saves a naive kernel program to a `.cu` file, which is then compiled and profiled by Nsight Compute (`ncu`) in the next cell."))
cells_to_add.append(nbf.v4.new_code_cell(prof_code))
cells_to_add.append(nbf.v4.new_code_cell("!nvcc prof_kernel.cu -o prof_kernel\n!ncu --set full ./prof_kernel"))


# Add plotting logic
plot_code = """import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re

# Since we don't have the outputs captured programmatically from Jupyter cells (without parsing the notebook JSON itself),
# we will provide a dictionary structure that you can update manually or programmatically.
# For illustration, here are the example values you might obtain running the cells above:

data = {
    'Kernel': [
        'Naive', 'GMEM Coalesce', 'SMEM Cache', 
        '1D Blocktiling', '2D Blocktiling', 'Vectorized', 
        'Autotuned', 'Warptiling'
    ],
    'GFLOPs': [
        300.0, 1900.0, 2900.0, 
        8400.0, 15900.0, 18200.0, 
        19700.0, 21700.0
    ]
}

df = pd.DataFrame(data)

plt.figure(figsize=(12, 6))
sns.barplot(data=df, x='GFLOPs', y='Kernel', palette='viridis')
plt.title('CUDA SGEMM Performance Optimization', fontsize=16)
plt.xlabel('Performance (GFLOPs)', fontsize=14)
plt.ylabel('Kernel Version', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()
"""

cells_to_add.append(nbf.v4.new_markdown_cell("### Plotting Performance Results\nUse the cell below to plot the performance data."))
cells_to_add.append(nbf.v4.new_code_cell(plot_code))

# Find the marker where we insert new cells. We will just insert them before the last cell if it's the markdown mentioning plotting.
insert_idx = len(nb.cells)
for i, cell in enumerate(nb.cells):
    if "profiling with nsight compute" in cell.source.lower():
        insert_idx = i
        break

# Insert the cells
nb.cells = nb.cells[:insert_idx] + cells_to_add

with open(nb_path, 'w') as f:
    nbf.write(nb, f)

print("Notebook updated successfully.")
