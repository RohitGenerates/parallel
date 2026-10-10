# Producer-Consumer OpenMP Simulator

## Description
This C program simulates a producer-consumer model where producers generate numerical computations and consumers execute a prime-number checking workload. It is designed to demonstrate multi-threading in **Shared Memory Systems via OpenMP**. 

We explore two concepts:
1. **Critical Sections (Ring Buffer):** A circular array buffer where `#pragma omp critical` is used to prevent race conditions during insertion and extraction.
2. **Scheduling Algorithms:** Testing the effect of `#pragma omp for schedule` with static, dynamic, and guided distribution of uneven computational tasks.

## Code Layout
`source/main.c` contains the primary C code with both tests.
`benchmark/` is the folder where the binary is compiled and stats are exported.

## Build and execution
To compile and run, you just need a bash terminal and gcc. Execute:
```bash
./build_and_run.sh
```
This will compile the C file, place the `.exe` in `benchmark`, execute it, print the tabular results to stdout, and output `benchmark_stats.tsv` (Tab-Separated Values).

## Findings
- **Producers vs Consumers:** Bottlenecks appear if there are too many producers and too few consumers (run out of buffer space) or vice versa (starvation).
- **Scheduling:** Dynamic scheduling often yields better uniform performance than static scheduling when tasks have unpredictable or highly skewed execution times because it self-balances instead of blindly slicing array chunks.
