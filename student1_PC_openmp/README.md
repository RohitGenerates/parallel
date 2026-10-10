# Network Packet Processor (OpenMP Producer-Consumer)

## Overview
This project simulates network packet processing using the OpenMP Shared Memory model. The architecture explores two things:
1. **Queue-based Tasking:** Multiple producers placing network packets into a shared bounded buffer, and multiple consumers competing to pop and process these packets. Synchronized using `omp_lock_t`.
2. **Loop Scheduling Imbalance:** Validation of `static`, `dynamic`, and `guided` scheduling on a batch of packets with heavy workload variations (to see which OpenMP scheduler handles imbalance best).

## Code Structure
- `src/main.cpp`: Main source code containing both `runQueueBasedPC()` and `runLoopScheduling()` functions.
- `Makefile`: Commands to compile the project.
- `scripts/plot_results.py`: Python script utilizing `pandas` and `matplotlib` to parse the output and generate visual performance charts.

## How to Compile & Run
1. Ensure `g++` and OpenMP are installed.
2. Compile and run the C++ simulation:
   ```bash
   make
   make run
   ```
   This will execute the benchmarks and generate a `results.csv` file.
3. Generate the graphs:
   ```bash
   python scripts/plot_results.py
   ```
   A `performance_analysis.png` file will be generated showing the evaluation.

## Methodology (For Report / Viva)
- **Locking:** `omp_set_lock` avoids false sharing or data races on the `std::queue`.
- **Uneven load:** The simulation purposefully sets half the packets to have a light workload, and half to have a massive workload.
- **Why dynamic wins:** Static chunk sizes fail because threads get stuck with heavy chunks, while dynamic automatically offloads packets to whichever thread becomes free first.
