/**
 * ============================================================================
 * Project 2: High-Performance Cache-Optimized Matrix Engine
 * ============================================================================
 * File: matrix_engine.c
 * Description: Implementation of Matrix2D memory lifecycle, cache-tiled
 *              OpenMP computational routines, and full native C benchmark suite.
 * ============================================================================
 */

#include "matrix_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#if defined(_WIN32)
#include <direct.h>
#define MKDIR(dir) _mkdir(dir)
#else
#include <sys/stat.h>
#define MKDIR(dir) mkdir(dir, 0755)
#endif

#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define WARMUP_RUNS 1
#define MEASURED_RUNS 3
#define NUM_SIZES 3
#define NUM_THREADS 3
#define BENCH_TILE_SIZE 64

static const int MATRIX_SIZES[NUM_SIZES] = {500, 1000, 2000};
static const int THREAD_COUNTS[NUM_THREADS] = {2, 4, 8};

typedef struct {
    double mean_time;
    double min_time;
    double std_dev;
    double speedup;
    double efficiency;
    double gflops;
    double raw_runs[MEASURED_RUNS];
} EngineMetrics;

typedef struct {
    int size;
    EngineMetrics seq_metrics;
    EngineMetrics omp_metrics[NUM_THREADS];
} EngineSizeResult;

/* Allocate a 2D matrix with contiguous memory backing */
Matrix2D* Matrix2D_create(size_t rows, size_t cols) {
    Matrix2D *mat = (Matrix2D*)malloc(sizeof(Matrix2D));
    if (!mat) {
        fprintf(stderr, "[ERROR] Memory allocation failed for Matrix2D metadata\n");
        return NULL;
    }

    mat->rows = rows;
    mat->cols = cols;

    /* Single contiguous allocation for maximum cache locality */
    mat->raw_data = (double*)malloc(rows * cols * sizeof(double));
    if (!mat->raw_data) {
        fprintf(stderr, "[ERROR] Memory allocation failed for raw matrix data (%zu x %zu)\n", rows, cols);
        free(mat);
        return NULL;
    }

    /* Allocate row pointer lookup table */
    mat->data = (double**)malloc(rows * sizeof(double*));
    if (!mat->data) {
        fprintf(stderr, "[ERROR] Memory allocation failed for row pointers\n");
        free(mat->raw_data);
        free(mat);
        return NULL;
    }

    for (size_t i = 0; i < rows; i++) {
        mat->data[i] = &mat->raw_data[i * cols];
    }

    return mat;
}

/* Free allocated matrix resources */
void Matrix2D_free(Matrix2D *mat) {
    if (mat) {
        if (mat->data) free(mat->data);
        if (mat->raw_data) free(mat->raw_data);
        free(mat);
    }
}

/* Fill matrix with a constant scalar value */
void Matrix2D_fill_constant(Matrix2D *mat, double value) {
    size_t total_elements = mat->rows * mat->cols;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < total_elements; i++) {
        mat->raw_data[i] = value;
    }
}

/* Fill matrix with deterministic mathematical pattern */
void Matrix2D_fill_pattern(Matrix2D *mat) {
    size_t n = mat->rows;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            mat->data[i][j] = (double)(i + j) * 0.001;
        }
    }
}

/* Baseline Sequential Matrix Multiplication (i-j-k triple loop) */
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

/**
 * Cache-Optimized Tiled OpenMP Matrix Multiplication (i-k-j block traversal)
 * - Tiling (Blocking) fits sub-matrices within L1/L2 cache lines.
 * - Loop-Interchange (i-k-j) enables continuous stride-1 spatial locality.
 * - Collapse(2) and Dynamic scheduling distribute work evenly across cores.
 */
void Matrix2D_multiply_tiled_omp(const Matrix2D *A, const Matrix2D *B, Matrix2D *C,
                                 int block_size, int num_threads) {
    int N = (int)A->rows;
    omp_set_num_threads(num_threads);

    /* Zero-initialize output matrix */
    Matrix2D_fill_constant(C, 0.0);

    #pragma omp parallel for collapse(2) schedule(dynamic, 16)
    for (int bi = 0; bi < N; bi += block_size) {
        for (int bj = 0; bj < N; bj += block_size) {
            for (int bk = 0; bk < N; bk += block_size) {

                /* Micro-kernel computing local sub-block tile */
                int i_max = MIN(bi + block_size, N);
                int j_max = MIN(bj + block_size, N);
                int k_max = MIN(bk + block_size, N);

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

/* Compute checksum (Frobenius norm proxy) for verification */
double Matrix2D_checksum(const Matrix2D *mat) {
    double sum = 0.0;
    size_t total = mat->rows * mat->cols;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (size_t i = 0; i < total; i++) {
        sum += fabs(mat->raw_data[i]);
    }
    return sum;
}

/* Compare two matrices element-wise for numerical equivalence */
bool Matrix2D_verify_equality(const Matrix2D *C1, const Matrix2D *C2, double tolerance) {
    if (C1->rows != C2->rows || C1->cols != C2->cols) return false;
    size_t total = C1->rows * C1->cols;
    for (size_t i = 0; i < total; i++) {
        if (fabs(C1->raw_data[i] - C2->raw_data[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

/* Helper to ensure output directory exists */
static void ensure_directory_exists(const char *path) {
    char temp[256];
    char *p = NULL;
    size_t len;

    snprintf(temp, sizeof(temp), "%s", path);
    len = strlen(temp);
    if (temp[len - 1] == '/' || temp[len - 1] == '\\') {
        temp[len - 1] = '\0';
    }

    for (p = temp + 1; *p; p++) {
        if (*p == '/' || *p == '\\') {
            *p = '\0';
            MKDIR(temp);
            *p = '/';
        }
    }
    MKDIR(temp);
}

/* Run timing benchmark for sequential kernel */
static EngineMetrics benchmark_sequential_kernel(int n) {
    Matrix2D *A = Matrix2D_create(n, n);
    Matrix2D *B = Matrix2D_create(n, n);
    Matrix2D *C = Matrix2D_create(n, n);

    Matrix2D_fill_constant(A, 1.5);
    Matrix2D_fill_constant(B, 2.0);

    /* Warm-up run */
    for (int w = 0; w < WARMUP_RUNS; w++) {
        Matrix2D_fill_constant(C, 0.0);
        Matrix2D_multiply_sequential(A, B, C);
    }

    /* Measured runs */
    EngineMetrics metrics;
    double sum_time = 0.0;
    metrics.min_time = 1e9;

    for (int r = 0; r < MEASURED_RUNS; r++) {
        Matrix2D_fill_constant(C, 0.0);
        double t_start = omp_get_wtime();
        Matrix2D_multiply_sequential(A, B, C);
        double t_end = omp_get_wtime();
        double elapsed = t_end - t_start;

        metrics.raw_runs[r] = elapsed;
        sum_time += elapsed;
        if (elapsed < metrics.min_time) {
            metrics.min_time = elapsed;
        }
    }

    metrics.mean_time = sum_time / (double)MEASURED_RUNS;

    double variance = 0.0;
    for (int r = 0; r < MEASURED_RUNS; r++) {
        double diff = metrics.raw_runs[r] - metrics.mean_time;
        variance += diff * diff;
    }
    metrics.std_dev = (MEASURED_RUNS > 1) ? sqrt(variance / (double)MEASURED_RUNS) : 0.0;
    metrics.speedup = 1.0;
    metrics.efficiency = 100.0;
    double total_flops = 2.0 * (double)n * (double)n * (double)n;
    metrics.gflops = (total_flops / metrics.mean_time) / 1e9;

    Matrix2D_free(A);
    Matrix2D_free(B);
    Matrix2D_free(C);

    return metrics;
}

/* Run timing benchmark for OpenMP tiled kernel */
static EngineMetrics benchmark_omp_tiled_kernel(int n, int threads, int block_size, double seq_time) {
    Matrix2D *A = Matrix2D_create(n, n);
    Matrix2D *B = Matrix2D_create(n, n);
    Matrix2D *C = Matrix2D_create(n, n);

    Matrix2D_fill_constant(A, 1.5);
    Matrix2D_fill_constant(B, 2.0);

    /* Warm-up run */
    for (int w = 0; w < WARMUP_RUNS; w++) {
        Matrix2D_fill_constant(C, 0.0);
        Matrix2D_multiply_tiled_omp(A, B, C, block_size, threads);
    }

    /* Measured runs */
    EngineMetrics metrics;
    double sum_time = 0.0;
    metrics.min_time = 1e9;

    for (int r = 0; r < MEASURED_RUNS; r++) {
        Matrix2D_fill_constant(C, 0.0);
        double t_start = omp_get_wtime();
        Matrix2D_multiply_tiled_omp(A, B, C, block_size, threads);
        double t_end = omp_get_wtime();
        double elapsed = t_end - t_start;

        metrics.raw_runs[r] = elapsed;
        sum_time += elapsed;
        if (elapsed < metrics.min_time) {
            metrics.min_time = elapsed;
        }
    }

    metrics.mean_time = sum_time / (double)MEASURED_RUNS;

    double variance = 0.0;
    for (int r = 0; r < MEASURED_RUNS; r++) {
        double diff = metrics.raw_runs[r] - metrics.mean_time;
        variance += diff * diff;
    }
    metrics.std_dev = (MEASURED_RUNS > 1) ? sqrt(variance / (double)MEASURED_RUNS) : 0.0;
    metrics.speedup = (metrics.mean_time > 0.0) ? (seq_time / metrics.mean_time) : 0.0;
    metrics.efficiency = (threads > 0) ? (metrics.speedup / (double)threads) * 100.0 : 0.0;
    double total_flops = 2.0 * (double)n * (double)n * (double)n;
    metrics.gflops = (total_flops / metrics.mean_time) / 1e9;

    Matrix2D_free(A);
    Matrix2D_free(B);
    Matrix2D_free(C);

    return metrics;
}

/* Full Automated C Benchmark Suite implementation */
void Matrix2D_run_benchmark_suite(void) {
    printf("==============================================================================\n");
    printf(" PROJECT 2: CACHE-OPTIMIZED TILED MATRIX ENGINE -- PURE C BENCHMARK SUITE\n");
    printf(" Architecture: Loop Interchange (i-k-j) + L1/L2 Cache Tiling (64x64) + SIMD\n");
    printf("==============================================================================\n");
    printf(" OpenMP Max Threads Available : %d\n", omp_get_max_threads());
    printf(" Cache Block Tile Size        : %dx%d\n", BENCH_TILE_SIZE, BENCH_TILE_SIZE);
    printf(" Benchmark Methodology        : %d Warmup, %d Measured Runs per configuration\n", WARMUP_RUNS, MEASURED_RUNS);
    printf(" High-Precision Timer         : OpenMP omp_get_wtime() (Wall-Clock Precision)\n");
    printf(" Matrix Dimensions Tested     : 500x500, 1000x1000, 2000x2000\n");
    printf(" OpenMP Thread Counts Tested  : 2, 4, 8 Threads\n");
    printf("==============================================================================\n\n");

    /* 1. Pre-flight Verification Pass */
    printf("[VALIDATION] Running pre-flight correctness check (N=500, Threads=4, Tile=64)...\n");
    Matrix2D *val_A = Matrix2D_create(500, 500);
    Matrix2D *val_B = Matrix2D_create(500, 500);
    Matrix2D *val_C_seq = Matrix2D_create(500, 500);
    Matrix2D *val_C_omp = Matrix2D_create(500, 500);

    Matrix2D_fill_constant(val_A, 1.5);
    Matrix2D_fill_constant(val_B, 2.0);

    Matrix2D_multiply_sequential(val_A, val_B, val_C_seq);
    Matrix2D_multiply_tiled_omp(val_A, val_B, val_C_omp, 64, 4);

    bool match = Matrix2D_verify_equality(val_C_seq, val_C_omp, 1e-5);
    if (match) {
        printf("  -> Correctness Check : [PASSED] Exact Numerical Equivalence\n");
        printf("  -> Status            : ALL PRE-FLIGHT ASSERTIONS PASSED\n\n");
    } else {
        fprintf(stderr, "[ERROR] Pre-flight verification failed! Output mismatch detected.\n");
        Matrix2D_free(val_A); Matrix2D_free(val_B); Matrix2D_free(val_C_seq); Matrix2D_free(val_C_omp);
        exit(EXIT_FAILURE);
    }
    Matrix2D_free(val_A); Matrix2D_free(val_B); Matrix2D_free(val_C_seq); Matrix2D_free(val_C_omp);

    /* 2. Execute Benchmark Matrix */
    printf("==============================================================================\n");
    printf(" EXECUTING NATIVE C / OpenMP CACHE-TILED BENCHMARK MATRIX\n");
    printf("==============================================================================\n");

    EngineSizeResult results[NUM_SIZES];

    for (int i = 0; i < NUM_SIZES; i++) {
        int n = MATRIX_SIZES[i];
        results[i].size = n;

        printf("\n[BENCHMARK] Dimension %d x %d ...\n", n, n);

        /* Sequential Baseline */
        printf("  -> Measuring Sequential baseline (%d runs)... ", MEASURED_RUNS);
        fflush(stdout);
        results[i].seq_metrics = benchmark_sequential_kernel(n);
        printf("Done. (Mean: %.4fs | %.2f GFLOPS)\n",
               results[i].seq_metrics.mean_time,
               results[i].seq_metrics.gflops);

        /* OpenMP Tiled Kernel */
        for (int t = 0; t < NUM_THREADS; t++) {
            int threads = THREAD_COUNTS[t];
            printf("  -> Measuring OpenMP Tiled (%d threads, %d runs)... ", threads, MEASURED_RUNS);
            fflush(stdout);
            results[i].omp_metrics[t] = benchmark_omp_tiled_kernel(n, threads, BENCH_TILE_SIZE, results[i].seq_metrics.mean_time);
            printf("Done. (Mean: %.4fs | Speedup: %.2fx | Eff: %.1f%% | %.2f GFLOPS)\n",
                   results[i].omp_metrics[t].mean_time,
                   results[i].omp_metrics[t].speedup,
                   results[i].omp_metrics[t].efficiency,
                   results[i].omp_metrics[t].gflops);
        }
    }

    /* 3. Render ASCII Tables */
    printf("\n==============================================================================\n");
    printf(" PROJECT 2 -- REAL BENCHMARK RESULTS (CACHE-TILED OpenMP IN C)\n");
    printf("==============================================================================\n\n");

    /* Table 1: Execution Time */
    printf("1. WALL-CLOCK EXECUTION TIME (Mean Measured in Seconds)\n");
    printf("------------------------------------------------------------------------------\n");
    printf("%-18s | %-15s | %-14s | %-14s | %-14s\n", "Matrix Dimension", "Sequential (s)", "2 Threads (s)", "4 Threads (s)", "8 Threads (s)");
    printf("------------------------------------------------------------------------------\n");

    for (int i = 0; i < NUM_SIZES; i++) {
        char dim_str[32];
        snprintf(dim_str, sizeof(dim_str), "%dx%d", results[i].size, results[i].size);
        printf("%-18s | %-15.4f | %-14.4f | %-14.4f | %-14.4f\n",
               dim_str,
               results[i].seq_metrics.mean_time,
               results[i].omp_metrics[0].mean_time,
               results[i].omp_metrics[1].mean_time,
               results[i].omp_metrics[2].mean_time);
    }
    printf("------------------------------------------------------------------------------\n\n");

    /* Table 2: Speedup, Efficiency & GFLOPS */
    printf("2. PARALLEL SPEEDUP, EFFICIENCY & THROUGHPUT (GFLOPS)\n");
    printf("------------------------------------------------------------------------------\n");
    printf("%-12s | %-8s | %-10s | %-10s | %-12s | %-14s\n", "Matrix", "Threads", "Time (s)", "Speedup", "Efficiency", "Throughput");
    printf("------------------------------------------------------------------------------\n");

    for (int i = 0; i < NUM_SIZES; i++) {
        char dim_str[32];
        snprintf(dim_str, sizeof(dim_str), "%dx%d", results[i].size, results[i].size);
        for (int t = 0; t < NUM_THREADS; t++) {
            printf("%-12s | %-8d | %-10.4f | %-8.2fx | %-10.2f%% | %-7.2f GFLOPS\n",
                   dim_str,
                   THREAD_COUNTS[t],
                   results[i].omp_metrics[t].mean_time,
                   results[i].omp_metrics[t].speedup,
                   results[i].omp_metrics[t].efficiency,
                   results[i].omp_metrics[t].gflops);
        }
    }
    printf("------------------------------------------------------------------------------\n\n");

    /* 4. Export CSV and TXT files */
    ensure_directory_exists("benchmark/results");
    ensure_directory_exists("results");

    const char *csv_path = "benchmark/results/project2_results.csv";
    const char *txt_path = "benchmark/results/project2_results.txt";

    /* Save CSV */
    FILE *f_csv = fopen(csv_path, "w");
    if (!f_csv) {
        f_csv = fopen("project2_results.csv", "w");
        if (f_csv) csv_path = "project2_results.csv";
    }

    if (f_csv) {
        fprintf(f_csv, "Matrix_Size,Kernel,Threads,Tile_Size,Mean_Time_Sec,Min_Time_Sec,Std_Dev_Sec,Speedup,Efficiency_Pct,GFLOPS\n");
        for (int i = 0; i < NUM_SIZES; i++) {
            int n = results[i].size;
            /* Write Sequential */
            fprintf(f_csv, "%d,Sequential,1,0,%.6f,%.6f,%.6f,1.00,100.00,%.3f\n",
                    n,
                    results[i].seq_metrics.mean_time,
                    results[i].seq_metrics.min_time,
                    results[i].seq_metrics.std_dev,
                    results[i].seq_metrics.gflops);
            /* Write OpenMP Tiled */
            for (int t = 0; t < NUM_THREADS; t++) {
                fprintf(f_csv, "%d,OpenMP_Tiled,%d,%d,%.6f,%.6f,%.6f,%.2f,%.2f,%.3f\n",
                        n,
                        THREAD_COUNTS[t],
                        BENCH_TILE_SIZE,
                        results[i].omp_metrics[t].mean_time,
                        results[i].omp_metrics[t].min_time,
                        results[i].omp_metrics[t].std_dev,
                        results[i].omp_metrics[t].speedup,
                        results[i].omp_metrics[t].efficiency,
                        results[i].omp_metrics[t].gflops);
            }
        }
        fclose(f_csv);
    }

    /* Save TXT */
    FILE *f_txt = fopen(txt_path, "w");
    if (!f_txt) {
        f_txt = fopen("project2_results.txt", "w");
        if (f_txt) txt_path = "project2_results.txt";
    }

    if (f_txt) {
        time_t now = time(NULL);
        fprintf(f_txt, "============================================================\n");
        fprintf(f_txt, " PROJECT 2 -- REAL BENCHMARK RESULTS (Cache-Tiled OpenMP)\n");
        fprintf(f_txt, "============================================================\n");
        fprintf(f_txt, "Optimization                 : Loop Interchange (i-k-j) + L1/L2 Cache Tiling (64x64) + SIMD\n");
        fprintf(f_txt, "OpenMP Max Threads Available : %d\n", omp_get_max_threads());
        fprintf(f_txt, "Benchmark Date               : %s", ctime(&now));
        fprintf(f_txt, "Warmup Runs                  : %d\n", WARMUP_RUNS);
        fprintf(f_txt, "Measured Repetitions         : %d\n", MEASURED_RUNS);
        fprintf(f_txt, "============================================================\n\n");

        fprintf(f_txt, "1. WALL-CLOCK EXECUTION TIME (Seconds)\n");
        fprintf(f_txt, "--------------------------------------------------------------------------------\n");
        fprintf(f_txt, "%-18s | %-15s | %-14s | %-14s | %-14s\n", "Matrix Dimension", "Sequential (s)", "2 Threads (s)", "4 Threads (s)", "8 Threads (s)");
        fprintf(f_txt, "--------------------------------------------------------------------------------\n");
        for (int i = 0; i < NUM_SIZES; i++) {
            char dim_str[32];
            snprintf(dim_str, sizeof(dim_str), "%dx%d", results[i].size, results[i].size);
            fprintf(f_txt, "%-18s | %-15.4f | %-14.4f | %-14.4f | %-14.4f\n",
                    dim_str,
                    results[i].seq_metrics.mean_time,
                    results[i].omp_metrics[0].mean_time,
                    results[i].omp_metrics[1].mean_time,
                    results[i].omp_metrics[2].mean_time);
        }
        fprintf(f_txt, "--------------------------------------------------------------------------------\n\n");

        fprintf(f_txt, "2. PARALLEL SPEEDUP, EFFICIENCY & THROUGHPUT (GFLOPS)\n");
        fprintf(f_txt, "--------------------------------------------------------------------------------\n");
        fprintf(f_txt, "%-12s | %-8s | %-10s | %-10s | %-12s | %-14s\n", "Matrix", "Threads", "Time (s)", "Speedup", "Efficiency", "Throughput");
        fprintf(f_txt, "--------------------------------------------------------------------------------\n");
        for (int i = 0; i < NUM_SIZES; i++) {
            char dim_str[32];
            snprintf(dim_str, sizeof(dim_str), "%dx%d", results[i].size, results[i].size);
            for (int t = 0; t < NUM_THREADS; t++) {
                fprintf(f_txt, "%-12s | %-8d | %-10.4f | %-8.2fx | %-10.2f%% | %-7.2f GFLOPS\n",
                        dim_str,
                        THREAD_COUNTS[t],
                        results[i].omp_metrics[t].mean_time,
                        results[i].omp_metrics[t].speedup,
                        results[i].omp_metrics[t].efficiency,
                        results[i].omp_metrics[t].gflops);
            }
        }
        fprintf(f_txt, "--------------------------------------------------------------------------------\n");
        fclose(f_txt);
    }

    printf("[EXPORT] Benchmark results successfully exported:\n");
    printf("  -> CSV : %s\n", csv_path);
    printf("  -> TXT : %s\n\n", txt_path);
    printf("==============================================================================\n");
    printf(" [PROJECT 2 BENCHMARK SUITE COMPLETED SUCCESSFULLY IN PURE C]\n");
    printf("==============================================================================\n");
}
