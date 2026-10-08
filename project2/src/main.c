/**
 * ============================================================================
 * Project 2: High-Performance Cache-Optimized Matrix Engine - Driver
 * ============================================================================
 * File: main.c
 * Description: Command-line interface and benchmark driver for Project 2.
 *
 * Compilation:
 *   gcc -O3 -fopenmp -static matrix_engine.c main.c -o matmul_engine -lm
 *
 * Usage:
 *   Run full benchmark suite:
 *     ./matmul_engine -B
 *     ./matmul_engine --benchmark
 *
 *   Run specific configuration:
 *     ./matmul_engine -n <size> -t <threads> -b <tile_size> -k <kernel> -v
 *     Example: ./matmul_engine -n 1000 -t 4 -b 64 -k omp_tiled
 * ============================================================================
 */

#include "matrix_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

void print_banner() {
    printf("======================================================================\n");
    printf(" PROJECT 2: CACHE-OPTIMIZED TILED MATRIX MULTIPLICATION (OpenMP)\n");
    printf(" Architecture: Loop Interchange (i-k-j) + L1/L2 Cache Tiling + SIMD\n");
    printf("======================================================================\n");
}

int main(int argc, char *argv[]) {
    int n = 1000;
    int threads = 4;
    int block_size = 64; // Tuned for standard 32KB/48KB L1 Data Cache
    char kernel[32] = "both"; // "seq", "omp_tiled", "both"
    int verify_mode = 0;
    int benchmark_mode = 0;

    /* Check if full benchmark mode requested */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-B") == 0 || strcmp(argv[i], "--benchmark") == 0 || strcmp(argv[i], "benchmark") == 0) {
            benchmark_mode = 1;
            break;
        }
    }

    if (benchmark_mode) {
        Matrix2D_run_benchmark_suite();
        return 0;
    }

    /* Parse standard CLI flags */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            block_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
            strncpy(kernel, argv[++i], sizeof(kernel) - 1);
            kernel[sizeof(kernel) - 1] = '\0';
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verify") == 0) {
            verify_mode = 1;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [-B | --benchmark] [-n size] [-t threads] [-b tile_size] [-k seq|omp_tiled|both] [-v]\n", argv[0]);
            printf("Options:\n");
            printf("  -B, --benchmark  Execute complete automated C benchmark suite (500, 1000, 2000)\n");
            printf("  -n <size>        Matrix dimension N x N (default: 1000)\n");
            printf("  -t <threads>     Number of OpenMP worker threads (default: 4)\n");
            printf("  -b <tile_size>   Cache block tile size (default: 64)\n");
            printf("  -k <kernel>      Kernel to execute: seq | omp_tiled | both (default: both)\n");
            printf("  -v, --verify     Verify numerical correctness between sequential and OpenMP tiled\n");
            return 0;
        }
    }

    if (n <= 0) n = 1000;
    if (threads <= 0) threads = 4;
    if (block_size <= 0) block_size = 64;

    print_banner();
    printf("Matrix Dimension (N x N)    : %d x %d\n", n, n);
    printf("Active OpenMP Threads       : %d\n", threads);
    printf("Cache Block Tile Size       : %d x %d\n", block_size, block_size);
    printf("Execution Kernel Mode       : %s\n", kernel);
    printf("Total Memory Footprint      : %.2f MB\n", (3.0 * n * n * sizeof(double)) / (1024.0 * 1024.0));
    printf("----------------------------------------------------------------------\n");

    /* Allocate matrices */
    Matrix2D *A = Matrix2D_create(n, n);
    Matrix2D *B = Matrix2D_create(n, n);
    Matrix2D *C_seq = NULL;
    Matrix2D *C_omp = NULL;

    if (!A || !B) {
        fprintf(stderr, "Error: Matrix initialization failed.\n");
        return 1;
    }

    Matrix2D_fill_constant(A, 1.5);
    Matrix2D_fill_constant(B, 2.0);

    double total_flops = 2.0 * (double)n * (double)n * (double)n;
    double seq_time = 0.0;
    double omp_time = 0.0;

    int run_seq = (strcmp(kernel, "seq") == 0 || strcmp(kernel, "naive") == 0 || strcmp(kernel, "both") == 0 || strcmp(kernel, "all") == 0 || verify_mode);
    int run_omp = (strcmp(kernel, "omp_tiled") == 0 || strcmp(kernel, "omp") == 0 || strcmp(kernel, "tiled") == 0 || strcmp(kernel, "both") == 0 || strcmp(kernel, "all") == 0 || verify_mode);

    if (run_seq) {
        C_seq = Matrix2D_create(n, n);
        if (!C_seq) {
            fprintf(stderr, "Error: C_seq allocation failed.\n");
            return 1;
        }
        printf("[1] Executing Sequential Baseline...\n");
        double t_seq_start = omp_get_wtime();
        Matrix2D_multiply_sequential(A, B, C_seq);
        double t_seq_end = omp_get_wtime();
        seq_time = t_seq_end - t_seq_start;
        double seq_gflops = (total_flops / seq_time) / 1e9;
        printf("    Sequential Runtime     : %.6f seconds\n", seq_time);
        printf("    Sequential Throughput  : %.3f GFLOPS\n", seq_gflops);
        printf("    Checksum (Seq)         : %.6e\n", Matrix2D_checksum(C_seq));
    }

    if (run_omp) {
        C_omp = Matrix2D_create(n, n);
        if (!C_omp) {
            fprintf(stderr, "Error: C_omp allocation failed.\n");
            return 1;
        }
        printf("[2] Executing Tiled OpenMP Kernel...\n");
        double t_omp_start = omp_get_wtime();
        Matrix2D_multiply_tiled_omp(A, B, C_omp, block_size, threads);
        double t_omp_end = omp_get_wtime();
        omp_time = t_omp_end - t_omp_start;
        double omp_gflops = (total_flops / omp_time) / 1e9;
        printf("    OpenMP Tiled Runtime   : %.6f seconds\n", omp_time);
        printf("    OpenMP Throughput      : %.3f GFLOPS\n", omp_gflops);
        printf("    Checksum (OpenMP)      : %.6e\n", Matrix2D_checksum(C_omp));
    }

    /* Comparison metrics if both executed */
    if (run_seq && run_omp) {
        double speedup = seq_time / omp_time;
        double efficiency = (speedup / (double)threads) * 100.0;
        printf("----------------------------------------------------------------------\n");
        printf(" PERFORMANCE EVALUATION METRICS:\n");
        printf("  -> Speedup Achieved        : %.2fx\n", speedup);
        printf("  -> Parallel Efficiency     : %.2f%%\n", efficiency);
        printf("  -> Compute Throughput      : %.3f GFLOPS\n", (total_flops / omp_time) / 1e9);

        bool match = Matrix2D_verify_equality(C_seq, C_omp, 1e-5);
        printf("  -> Correctness Check       : %s\n", match ? "[PASSED] Exact Numerical Equivalence" : "[FAILED] Output Mismatch");
        if (!match) {
            fprintf(stderr, "Validation failed: Output mismatch between sequential and OpenMP tiled\n");
        }
    }

    printf("======================================================================\n");

    /* Cleanup */
    Matrix2D_free(A);
    Matrix2D_free(B);
    if (C_seq) Matrix2D_free(C_seq);
    if (C_omp) Matrix2D_free(C_omp);

    return 0;
}
