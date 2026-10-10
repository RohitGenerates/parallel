# Report Template: Prime Number OpenMP Simulator

## 1. Introduction
The **Producer-Consumer** algorithm is considered one of the classic synchronization structures in distributed / parallel computing. This project focuses on verifying simulated prime-numbers by computing mathematical workloads inside a multi-threaded Shared Memory environment powered by **OpenMP**.

**Objective:**
Implement a fixed Ring Buffer (circular queue) populated by producer threads and read by consumer threads, enforcing thread safety via OpenMP critical regions. We will also benchmark OpenMP `#pragma omp for schedule()` variations to analyze thread execution behaviors.

## 2. Problem Statement
Implement the simulation in:
- Standard `C` using `#pragma omp critical` for shared buffer access.
- Calculate speedups for standard Producer/Consumer ratios: `2x2`, `4x1`, `1x4`.
- Validate whether `dynamic` scheduling outperforms `static` scheduling during skewed calculations.

## 3. Algorithm Snippet Layout
**Critical Section Implementation:**
Insertions into the ring buffer are bound inside `critical` sections, guaranteeing that multiple producer threads cannot maliciously edit the `tail` pointer simultaneously, inherently preventing data-races.
```c
#pragma omp critical
{
    if (count < MAX_BUFFER_SIZE) {
        task_buffer[tail] = new_item;
        tail = (tail + 1) % MAX_BUFFER_SIZE;
        count++;
    }
}
```

## 4. Experimental Setup
**System Specifications:**
- **OS / Compiler:** Windows MinGW, GCC `16.2.0`, `-fopenmp` flag.
- **Language:** Standard C.
- **Tasks Computed:** 40,000 algorithmic cycles.

## 5. Results Table (Execution Time Extraction)
*(Extract tabular results from `benchmark_stats.tsv`)*

**Phase 1: Thread Mapping Evaluation:**
| Configuration (P-C) | Time (Seconds) | Analysis |
|---------------------|----------------|----------|
| 1 Producer, 1 Cons. | 0.10s          | Baseline. |
| 2 Producer, 2 Cons. | 0.05s          | Noticeable ~2x speedup. |
| 1 Producer, 4 Cons. | 0.02s          | Heavy consumer parallelization handles pipeline fast. |
| 4 Producer, 1 Cons. | 0.13s          | Consumer starvation prevents improvement. |

**Phase 2: Scheduling Heuristics (40K tasks):**
| Max Threads | Static Schedule | Dynamic Schedule | 
|-------------|-----------------|------------------|
| 2 Threads   | 0.024s          | 0.024s           |
| 4 Threads   | 0.013s          | 0.012s           |
| 8 Threads   | 0.008s          | 0.007s           |

## 6. Analysis & Discussion
- **False Sharing / Memory Overhead**: Unlike distributed topologies (like MPI), OpenMP requires specific safeguards (`critical` / `atomic`) over scalar counters (`head`, `tail`). Failure to enforce these locks breaks indices.
- **Scheduling**: Dynamic threading achieved a slightly higher throughput (lower execution time) than static scheduling because it actively reallocates complex prime computations to currently sleeping/freed threads.

## 7. Conclusion
Shared Memory OpenMP environments natively accelerate computationally heavy models like standard Primality checkers or Math integrators. The results definitively illustrate that aligning the Producer and Consumer thread ratios correctly guarantees higher performance and circumvents wait-time pipeline bottlenecking.
