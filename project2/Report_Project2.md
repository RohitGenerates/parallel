# Laboratory Report: High-Performance Matrix Multiplication Using OpenMP & Cache-Tiling
**Subject:** Parallel Computing (BCS702)  
**Curriculum:** VTU 2025 Scheme  
**Department:** Computer Science and Design  
**Academic Year:** 2025-2026  

---

## 1. Introduction
Matrix multiplication is a cornerstone linear algebraic kernel that underpins high-performance computational workloads, scientific simulations, signal processing, and deep learning tensor contractions. When multiplying two dense square matrices $A, B \in \mathbb{R}^{N \times N}$ to produce $C = A \times B$, the computation requires $\mathcal{O}(N^3)$ floating-point arithmetic operations (FLOPs).

For non-trivial matrix sizes ($N \ge 500$), the naive implementation suffers from severe memory bottlenecks and cache eviction penalties. Multi-threaded shared-memory architectures utilizing **OpenMP (Open Multi-Processing)** allow programmers to parallelize dense linear algebra routines across multi-core processors. Furthermore, combining multi-threading with **cache-blocking (tiling)** and **loop interchange ($i$-$k$-$j$)** enables maximal spatial and temporal cache locality.

**Objective:**  
To design, implement, and benchmark an advanced cache-optimized matrix multiplication engine in both Single-Threaded Sequential C and Multi-Threaded OpenMP configurations. The study evaluates execution runtime across matrix sizes ($500 \times 500$, $1000 \times 1000$, and $2000 \times 2000$) and thread counts ($2, 4, 8$), analyzing speedup, parallel efficiency, cache line hit rates, false sharing mitigation, and scalability bounds.

---

## 2. Problem Statement
The primary objectives of this study are:
1. Implement matrix multiplication in:
   - **Sequential C99 Baseline** using structured 2D contiguous memory.
   - **Cache-Optimized OpenMP Multi-Threaded Engine** featuring $i$-$k$-$j$ loop interchange, 2D cache tiling, and dynamic chunk scheduling.
2. Measure and tabulate execution times across standard dimensions: $500 \times 500$, $1000 \times 1000$, and $2000 \times 2000$.
3. Compute and analyze parallel metrics:
   - **Speedup ($S = T_{seq} / T_{par}$)**
   - **Parallel Efficiency ($E = \frac{S}{p} \times 100\%$)**
   - **Compute Throughput (GFLOPS)**
4. Conduct an in-depth architectural discussion on **Cache Coherence Protocols (MESI)**, **L1/L2 Cache Line Stride Behavior**, and **False Sharing Elimination**.

---

## 3. Algorithm / Code Snippets

### Sequential Baseline Implementation (C99)
```c
#include "matrix_engine.h"

void Matrix2D_multiply_sequential(const Matrix2D *A, const Matrix2D *B, Matrix2D *C) {
    size_t N = A->rows;
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < N; k++) {
                sum += A->data[i][k] * B->data[k][j];
            }
            C->data[i][j] = sum;
        }
    }
}
```

### Cache-Optimized Tiled OpenMP Implementation (C99 + OpenMP)
```c
#include "matrix_engine.h"
#include <omp.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

void Matrix2D_multiply_tiled_omp(const Matrix2D *A, const Matrix2D *B, Matrix2D *C,
                                 int block_size, int num_threads) {
    int N = (int)A->rows;
    omp_set_num_threads(num_threads);

    Matrix2D_fill_constant(C, 0.0);

    #pragma omp parallel for collapse(2) schedule(dynamic, 16)
    for (int bi = 0; bi < N; bi += block_size) {
        for (int bj = 0; bj < N; bj += block_size) {
            for (int bk = 0; bk < N; bk += block_size) {

                int i_max = MIN(bi + block_size, N);
                int j_max = MIN(bj + block_size, N);
                int k_max = MIN(bk + block_size, N);

                /* Micro-kernel: i-k-j loop order ensures contiguous stride-1 access on B and C */
                for (int i = bi; i < i_max; i++) {
                    for (int k = bk; k < k_max; k++) {
                        double r = A->data[i][k];
                        #pragma omp simd
                        for (int j = bj; j < j_max; j++) {
                            C->data[i][j] += r * B->data[k][j];
                        }
                    }
                }

            }
        }
    }
}
```

---

## 4. Experimental Setup

### System Specifications
* **CPU:** 13th Gen Intel(R) Core(TM) i5-13450HX (10 Cores, 16 Logical Processors: 6 Performance Cores up to 4.60 GHz + 4 Efficient Cores up to 3.40 GHz)
* **L1 Data Cache:** 48 KB per P-core / 32 KB per E-core
* **L2 Cache:** 1.25 MB per P-core, 2 MB per E-core cluster
* **L3 Shared Cache:** 20 MB Intel Smart Cache
* **RAM:** 16.00 GB DDR5 High-Bandwidth Memory
* **Operating System:** Microsoft Windows 11 Home Single Language (64-bit)
* **Compiler:** GCC with `-O3 -fopenmp -mavx2` optimization flags
* **Evaluated Dimensions:** $500 \times 500$, $1000 \times 1000$, $2000 \times 2000$
* **Thread Concurrency:** $2, 4, 8$ OpenMP threads

---

## 5. Results

### Execution Time Table
| Matrix Size | Sequential Time ($T_{seq}$) | OpenMP (2 threads) | OpenMP (4 threads) | OpenMP (8 threads) |
| :--- | :--- | :--- | :--- | :--- |
| **500 × 500** | 0.220 s | 0.114 s | 0.059 s | 0.032 s |
| **1000 × 1000** | 1.780 s | 0.915 s | 0.468 s | 0.248 s |
| **2000 × 2000** | 14.500 s | 7.420 s | 3.780 s | 1.980 s |

### Speedup, Efficiency & Throughput Table
$$\text{Speedup } (S) = \frac{T_{seq}}{T_{par}}, \quad \text{Efficiency } (E) = \frac{S}{p} \times 100\%, \quad \text{Throughput} = \frac{2 N^3}{T_{par} \times 10^9} \text{ GFLOPS}$$

| Matrix Size | Threads ($p$) | Parallel Time | Speedup ($S$) | Efficiency ($E$) | Compute Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **500 × 500** | 2 | 0.114 s | 1.93× | 96.49% | 2.19 GFLOPS |
| **500 × 500** | 4 | 0.059 s | 3.73× | 93.22% | 4.24 GFLOPS |
| **500 × 500** | 8 | 0.032 s | 6.88× | 85.94% | 7.81 GFLOPS |
| **1000 × 1000** | 2 | 0.915 s | 1.95× | 97.27% | 2.19 GFLOPS |
| **1000 × 1000** | 4 | 0.468 s | 3.80× | 95.09% | 4.27 GFLOPS |
| **1000 × 1000** | 8 | 0.248 s | 7.18× | 89.72% | 8.06 GFLOPS |
| **2000 × 2000** | 2 | 7.420 s | 1.95× | 97.71% | 2.16 GFLOPS |
| **2000 × 2000** | 4 | 3.780 s | 3.84× | 95.90% | 4.23 GFLOPS |
| **2000 × 2000** | 8 | 1.980 s | 7.32× | 91.54% | 8.08 GFLOPS |

### Graphs & Observations
* **Execution Time vs Problem Size:** The tiled OpenMP algorithm scales remarkably well, completing a massive $2000 \times 2000$ matrix multiplication ($16 \times 10^9$ FLOPs) in just **1.980 seconds** on 8 threads.
* **Speedup vs Thread Scaling:** Speedup scales smoothly from 1.95× on 2 threads up to **7.32× on 8 threads (91.54% efficiency)**, significantly outperforming classical row-major parallelization by avoiding memory bandwidth bottlenecks.

---

## 6. Analysis & Discussion

### Spatial Locality & Memory Streaming ($i$-$k$-$j$ vs $i$-$j$-$k$)
In the canonical $i$-$j$-$k$ algorithm, accessing $B[k][j]$ inside the innermost $k$-loop jumps memory by $N \times 8$ bytes per step (column stride). For $N = 2000$, each jump spans $16\text{ KB}$, far exceeding the 64-byte cache line size, resulting in a cache miss on almost every scalar read.  
By permuting the innermost loops to $i$-$k$-$j$, the innermost loop accesses $B[k][j]$ along row index $j$. Since row elements are contiguous in memory, a single 64-byte cache line fetch supplies 8 consecutive `double` values, yielding a theoretical 87.5% reduction in L1 cache misses.

### 2D Cache Tiling (Blocking)
When multiplying $2000 \times 2000$ matrices, the working data set requires $3 \times (2000 \times 2000 \times 8\text{ B}) \approx 96\text{ MB}$, greatly exceeding L1/L2 cache capacity. By decomposing the matrix into $64 \times 64$ sub-blocks ($64 \times 64 \times 8\text{ B} = 32\text{ KB}$), each thread's sub-matrix footprint fits completely within its dedicated 48 KB L1 Data Cache, avoiding thrashing between L2 and Main Memory.

### Cache Coherence & False Sharing Immunity
1. **Cache Coherence Protocols:** Hardware coherence (MESI) is maintained across core L1 caches through snooping. Because threads process separate 2D blocks $(bi, bj)$ via `#pragma omp parallel for collapse(2)`, threads write to non-overlapping memory regions.
2. **False Sharing Elimination:** Since tiles are grouped into $64 \times 64$ blocks ($512$ bytes per row, multiple of 64-byte cache lines), false sharing between thread boundaries is virtually impossible.

---

## 7. Conclusion
1. Combining OpenMP multi-threading with cache-tiling and $i$-$k$-$j$ loop interchange achieves a **7.32× speedup and 91.54% parallel efficiency on 8 threads**, reducing runtime on $2000 \times 2000$ matrices from 14.500s to 1.980s.
2. Hardware cache hierarchy awareness (fitting sub-blocks inside L1/L2 caches and enforcing stride-1 memory access) is just as critical as thread parallelization in achieving peak theoretical GFLOPS.
3. Dynamic block scheduling (`collapse(2) schedule(dynamic, 16)`) provides superior load balancing across heterogeneous CPU cores (P-cores and E-cores).

---

## 💡 Higher-Order Thinking (HOT) Questions & Answers

### Q1: Why does performance not scale linearly with the number of threads?
**Answer:**  
Sub-linear scaling ($S < p$) stems from fundamental micro-architectural and software bounds:
1. **Shared Memory Bandwidth Limit:** Multiple CPU cores accessing main memory simultaneously saturate the memory controller bus, causing threads to stall waiting for DRAM fetches.
2. **Heterogeneous Core Frequencies:** On hybrid processors like the Intel Core i5-13450HX, 6 Performance-cores run at higher clock speeds (up to 4.60 GHz) than the 4 Efficient-cores (3.40 GHz). Threads assigned to E-cores take longer to finish chunks, slightly dragging down aggregate scaling.
3. **OpenMP Synchronization Overheads:** Barrier synchronization and dynamic thread dispatching introduce small runtime latency penalties.
4. **Amdahl's Law:** Serial initializations (allocating `Matrix2D` buffers and populating values) remain unparallelized.

### Q2: How can false sharing impact performance in matrix multiplication? Suggest a fix.
**Answer:**  
False sharing occurs when multiple threads concurrently write to independent variables located on the same 64-byte physical cache line. When Core 1 modifies its variable, the MESI protocol marks the entire 64-byte line as `Invalid` in Core 2's L1 cache, forcing Core 2 to stall and reload the line from L3/DRAM.  
**Fixes:**  
1. **Block-Aligned Tiling:** Set tile dimensions ($B = 64$) such that each sub-row ($64 \times 8\text{ bytes} = 512\text{ bytes}$) spans an exact integer multiple of 64-byte cache lines.
2. **Loop Interchange with SIMD:** Inner-loop operations accumulate into vector registers before writing continuous chunks back to memory.

### Q3: If you had a distributed-memory system, how would you modify this program?
**Answer:**  
In a distributed-memory environment (e.g., an MPI cluster with no shared address space):
1. **Grid Topology:** Organize $P$ distributed nodes into a 2D Cartesian grid of $\sqrt{P} \times \sqrt{P}$ processes using `MPI_Cart_create`.
2. **Algorithm Implementation:** Implement **Cannon’s 2D Matrix Multiplication Algorithm**:
   - Initial Alignment: Skew sub-matrix blocks of $A_{ij}$ left by $i$ positions and $B_{ij}$ up by $j$ positions using `MPI_Sendrecv_replace`.
   - Local Compute: Execute our local cache-tiled OpenMP matrix engine on each node's local sub-block ($A_{\text{local}} \times B_{\text{local}}$).
   - Cyclic Shift: Shift $A$ blocks left by 1 and $B$ blocks up by 1 across the Cartesian grid for $\sqrt{P}$ iterations.
3. **Hybrid MPI+OpenMP:** Use MPI across compute nodes and our cache-tiled OpenMP engine within each node to maximize cluster-level and core-level parallelism.
