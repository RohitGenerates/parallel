# High-Performance Parallel Matrix Multiplication Suite (C & OpenMP)

This repository contains two comprehensive Parallel Computing laboratory projects demonstrating multi-core parallelization, cache hierarchy optimization, SIMD vectorization, and performance profiling in **100% pure C and OpenMP**.

* **`project1/`** — **Standard Row-Major Parallel Matrix Multiplication (Sequential vs OpenMP)**
* **`project2/`** — **Cache-Optimized Tiled Matrix Multiplication Engine (OpenMP, Loop-Interchange & SIMD)**

All benchmarks compile to native standalone C binaries and execute directly on the host system to collect real wall-clock timings (`omp_get_wtime()`) and compute empirical speedup, parallel efficiency, and GFLOPS throughput—writing structured tables and reports directly via standard C file I/O.

---

## 🛠️ Prerequisites & Compiler Setup

### 1. GCC with OpenMP Support

#### On Linux (Ubuntu / Debian / Lab Systems):
```bash
sudo apt update
sudo apt install -y build-essential
```

#### On Windows (PowerShell / Command Prompt):
Install MinGW-w64 with OpenMP support via `winget`:
```powershell
winget install --id BrechtSanders.WinLibs.POSIX.UCRT
```

#### Verify Toolchain:
```bash
gcc --version
```

---

## 🚀 Quickstart: Build & Run All Benchmarks

You can build and execute both project benchmarks from the repository root using Make:

```bash
# 1. Compile all C binaries across Project 1 and Project 2
make all

# 2. Run both native C benchmark suites sequentially
make run
```

---

## 📁 Repository Structure
```text
.
├── Makefile                    # Root Makefile to build and run both projects
├── README.md                   # Repository-wide overview and instructions
├── project1/                   # Project 1: Standard Row-Major OpenMP
│   ├── Makefile                # Project 1 build configuration
│   ├── src/
│   │   ├── matmul_seq.c        # Sequential matrix multiplication baseline in C
│   │   ├── matmul_omp.c        # OpenMP parallel matrix multiplication in C
│   │   └── benchmark.c         # Pure C automated benchmark suite & reporter
│   ├── bin/                    # Compiled native executables
│   │   ├── matmul_seq
│   │   ├── matmul_omp
│   │   └── benchmark
│   ├── benchmark/
│   │   └── results/            # Measured real benchmark outputs (CSV, TXT)
│   │       ├── project1_results.csv
│   │       └── project1_results.txt
│   ├── Report_Project1.docx    # Formatted Word laboratory report
│   ├── Report_Project1.md      # Markdown laboratory report
│   └── README.md               # Project 1 documentation
└── project2/                   # Project 2: Cache-Tiled OpenMP Engine
    ├── Makefile                # Project 2 build configuration
    ├── src/
    │   ├── matrix_engine.h     # Matrix2D abstraction & kernel prototypes
    │   ├── matrix_engine.c     # Tiled OpenMP & sequential algorithms
    │   └── main.c              # CLI driver & pure C automated benchmark suite
    ├── bin/                    # Compiled native executables
    │   └── matmul_engine
    ├── benchmark/
    │   └── results/            # Measured real benchmark outputs (CSV, TXT)
    │       ├── project2_results.csv
    │       └── project2_results.txt
    ├── Report_Project2.docx    # Formatted Word laboratory report
    ├── Report_Project2.md      # Markdown laboratory report
    └── README.md               # Project 2 documentation
```

---

## 🔬 Individual Project Execution

### Project 1: Standard Row-Major Matrix Multiplication
```bash
cd project1

# Run full C benchmark suite:
make run

# Or compile and run individual binaries:
gcc -O3 -Wall -fopenmp -static src/benchmark.c -o bin/benchmark -lm
./bin/benchmark

# Run single sequential or OpenMP configuration:
./bin/matmul_seq 1000
./bin/matmul_omp 1000 4
```

### Project 2: Cache-Optimized Tiled Matrix Engine
```bash
cd project2

# Run full C benchmark suite:
make run

# Or compile and run standalone CLI:
gcc -O3 -Wall -Wextra -static -fopenmp src/matrix_engine.c src/main.c -o bin/matmul_engine -lm
./bin/matmul_engine --benchmark

# Run single configuration with custom parameters:
./bin/matmul_engine -n 1000 -t 4 -b 64 -k both
./bin/matmul_engine -n 2000 -t 8 -b 64 -k omp_tiled
./bin/matmul_engine -n 500 -t 4 -b 64 -v
```

---

## 📊 Performance Metrics & Formulas

### 1. Speedup ($S_p$)
Measures execution time reduction using $p$ parallel threads relative to single-threaded baseline:
$$S_p = \frac{T_{\text{sequential}}}{T_{\text{parallel}}(p)}$$

### 2. Parallel Efficiency ($E_p$)
Measures the percentage of theoretical compute scaling achieved per core:
$$E_p = \left( \frac{S_p}{p} \right) \times 100\%$$

### 3. Arithmetic Throughput (GFLOPS)
For dense square matrix multiplication ($N \times N$), total operations are $2N^3$ floating-point ops:
$$\text{GFLOPS} = \frac{2 \times N^3}{\text{Execution Time (seconds)} \times 10^9}$$

---

## 🔬 Architectural Comparison: Project 1 vs. Project 2

| Feature | Project 1 (Row-Major OpenMP) | Project 2 (Cache-Tiled OpenMP Engine) |
| :--- | :--- | :--- |
| **Algorithmic Complexity** | $\mathcal{O}(N^3)$ | $\mathcal{O}(N^3)$ |
| **Inner Loop Order** | $i \to j \to k$ | $i \to k \to j$ (Loop Interchange) |
| **Memory Access Pattern** | Column jumps across Matrix $B$ (stride-$N$) | Contiguous row streaming on Matrix $B$ (stride-1) |
| **Cache Optimization** | Relies on L3/hardware prefetcher | 2D Cache Tiling ($64 \times 64$) fitted to L1/L2 caches |
| **Vectorization** | Hindered by stride-$N$ memory access | `#pragma omp simd` vectorization enabled |
| **OpenMP Scheduling** | Static 1D row chunking (`schedule(static)`) | 2D tile block collapse (`collapse(2) schedule(dynamic)`) |
| **Observed $2000 \times 2000$ Speedup** | $\approx 5.59\times$ on 8 threads | $\approx 60.33\times$ on 8 threads |
| **Observed $2000 \times 2000$ Throughput** | $\approx 5.21\text{ GFLOPS}$ | $\approx 56.01\text{ GFLOPS}$ |
