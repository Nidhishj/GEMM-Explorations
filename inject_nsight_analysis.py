import json

notebook_path = '/ssd_scratch/nidhishj/GEMM-Explorations/GEMM_Assignment_T4.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

# The cells where we ran `!ncu` are roughly at indices 26, 30, and 33.
# We'll just locate them by string matching in the source.

new_cells = []
for cell in nb['cells']:
    new_cells.append(cell)
    
    if cell['cell_type'] == 'code' and any('prof_kernel2' in line and '!ncu' in line for line in cell['source']):
        explanation = {
          "cell_type": "markdown",
          "metadata": {},
          "source": [
            "### Nsight Compute Analysis: Naive vs. Global Memory Coalescing\\n",
            "\\n",
            "By comparing the Nsight Compute (`ncu`) results of **Kernel 1** and **Kernel 2**, we can observe the dramatic impact of memory access patterns:\\n",
            "\\n",
            "1. **Naive Kernel (Kernel 1)**: The profiling report shows a severe bottleneck in memory bandwidth. Because threads read non-contiguous global memory addresses, the accesses are **uncoalesced**. This thrashes the cache, leading to a low **L1/TEX Cache Throughput (16.5%)** and forcing the GPU to repeatedly fetch from slow DRAM (Duration: ~2362 ms). Consequently, the Compute (SM) units starve for data, resulting in a very low compute utilization (~22%).\\n",
            "2. **Global Memory Coalescing (Kernel 2)**: By simply remapping our grid/block layout so that adjacent threads (`threadIdx.x`) access adjacent memory elements (`cCol`), we achieve **coalesced memory transactions**. The `ncu` metrics reflect this perfectly: **L1/TEX Cache Throughput rockets to ~87%**, and execution duration plummets to ~422 ms (a 5.5x speedup!). Compute (SM) Throughput also improves massively to ~87%.\\n",
            "\\n",
            "**The New Bottleneck:** Despite the speedup, Kernel 2 is now bottlenecked by **Memory Throughput (~87%)**. Every element of `A` and `B` is still fetched from global memory $K$ times. To push performance further and reach cuBLAS levels, we must eliminate redundant global memory reads by caching tiles of data into fast **Shared Memory**."
          ]
        }
        new_cells.append(explanation)
        
    elif cell['cell_type'] == 'code' and any('prof_cublas' in line and '!ncu' in line for line in cell['source']):
        explanation = {
          "cell_type": "markdown",
          "metadata": {},
          "source": [
            "### Nsight Compute Analysis: cuBLAS Speed-of-Light\\n",
            "\\n",
            "The profiling results for NVIDIA's highly optimized **cuBLAS** reference implementation reveal what a fully compute-bound kernel looks like:\\n",
            "\\n",
            "- **Compute (SM) Throughput** reaches **~96.6%**, meaning the multiprocessors are almost constantly crunching math without stalling.\\n",
            "- **Memory Throughput** drops to a comfortable ~42.5%, and DRAM Throughput is minimal (~14.7%).\\n",
            "- **Duration** is a blazing-fast ~47.5 ms (almost 10x faster than our coalesced kernel!).\\n",
            "\\n",
            "**How does cuBLAS achieve this?**\\n",
            "Instead of relying purely on the L1 cache, cuBLAS employs advanced optimization strategies: **Shared Memory Tiling** (loading chunks of `A` and `B` into shared memory), **Register Tiling** (accumulating results directly in registers), **Thread Coarsening** (each thread computes multiple elements of `C`), and **Vectorized Memory Accesses** (e.g., `float4`). The rest of this notebook will guide you through implementing these exact techniques to close the gap with cuBLAS!"
          ]
        }
        new_cells.append(explanation)

nb['cells'] = new_cells

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=2)

print('Successfully injected Nsight Compute analysis cells.')
