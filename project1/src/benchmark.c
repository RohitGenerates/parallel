/**
 * ============================================================================
 * Project 1: Pure C Automated Benchmark & Performance Evaluation Suite
 * ============================================================================
 * File: benchmark.c
 * Description: 100% Native C/OpenMP benchmark driver evaluating Standard
 *              Row-Major Matrix Multiplication across dimensions (500, 1000, 2000)
 *              and thread counts (2, 4, 8).
 *              Measures real wall-clock execution time with omp_get_wtime(),
 *              computes statistical metrics (mean, min, std dev, speedup, efficiency),
 *              renders formatted ASCII tables, and exports CSV and TXT reports
 *              directly using standard C file I/O.
 *
 * Compilation:
 *   gcc -O3 -fopenmp benchmark.c -o benchmark -lm
 *
 * Usage:
 *   ./benchmark
 * ============================================================================
 */

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

#define WARMUP_RUNS 1
#define MEASURED_RUNS 3
#define NUM_SIZES 3
#define NUM_THREADS 3

static const int MATRIX_SIZES[NUM_SIZES] = {500, 1000, 2000};
static const int THREAD_COUNTS[NUM_THREADS] = {2, 4, 8};

typedef struct {
    double mean_time;
    double min_time;
    double std_dev;
    double speedup;
    double efficiency;
    double raw_runs[MEASURED_RUNS];
} PerfMetrics;

typedef struct {
    int size;
    PerfMetrics seq_metrics;
    PerfMetrics omp_metrics[NUM_THREADS];
} SizeBenchmarkResult;

/* Allocate 1D flattened matrix */
double* allocate_matrix(int n) {
    double *mat = (double*)malloc((size_t)n * n * sizeof(double));
    if (!mat) {
        fprintf(stderr, "[ERROR] Memory allocation failed for size %d x %d\n", n, n);
        exit(EXIT_FAILURE);
    }
    return mat;
}

/* Initialize matrix */
void initialize_matrix(double *mat, int n, double val) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n * n; i++) {
        mat[i] = val;
    }
}

/* Sequential Matrix Multiplication */
void matrix_multiply_seq(const double *A, const double *B, double *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

/* OpenMP Parallel Matrix Multiplication */
void matrix_multiply_omp(const double *A, const double *B, double *C, int n, int num_threads) {
    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0; // Thread-private accumulation register (eliminates false sharing)
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

/* Compute checksum of matrix */
double compute_checksum(const double *mat, int n) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (int i = 0; i < n * n; i++) {
        sum += fabs(mat[i]);
    }
    return sum;
}

/* Ensure directory exists */
void ensure_directory_exists(const char *path) {
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

/* Pre-flight correctness validation check */
void run_preflight_validation(void) {
    int test_n = 500;
    printf("[VALIDATION] Running pre-flight correctness check (N=%d, Threads=4)...\n", test_n);

    double *A = allocate_matrix(test_n);
    double *B = allocate_matrix(test_n);
    double *C_seq = allocate_matrix(test_n);
    double *C_omp = allocate_matrix(test_n);

    initialize_matrix(A, test_n, 1.5);
    initialize_matrix(B, test_n, 2.0);
    initialize_matrix(C_seq, test_n, 0.0);
    initialize_matrix(C_omp, test_n, 0.0);

    matrix_multiply_seq(A, B, C_seq, test_n);
    matrix_multiply_omp(A, B, C_omp, test_n, 4);

    double expected_sample = 1.5 * 2.0 * (double)test_n;
    int seq_pass = 1;
    int omp_pass = 1;

    for (int i = 0; i < test_n * test_n; i += (test_n + 1)) {
        if (fabs(C_seq[i] - expected_sample) > 1e-5) seq_pass = 0;
        if (fabs(C_omp[i] - expected_sample) > 1e-5) omp_pass = 0;
    }

    if (seq_pass && omp_pass) {
        printf("  -> Sequential Baseline Check : [PASSED] Sample C[0]=%.2f (Expected: %.2f)\n", C_seq[0], expected_sample);
        printf("  -> OpenMP Parallel Check     : [PASSED] Sample C[0]=%.2f (Expected: %.2f)\n", C_omp[0], expected_sample);
        printf("  -> Status                    : ALL PRE-FLIGHT ASSERTIONS PASSED\n\n");
    } else {
        fprintf(stderr, "[ERROR] Pre-flight verification failed!\n");
        free(A); free(B); free(C_seq); free(C_omp);
        exit(EXIT_FAILURE);
    }

    free(A);
    free(B);
    free(C_seq);
    free(C_omp);
}

/* Run timing benchmark for sequential algorithm */
PerfMetrics benchmark_sequential(int n) {
    double *A = allocate_matrix(n);
    double *B = allocate_matrix(n);
    double *C = allocate_matrix(n);

    initialize_matrix(A, n, 1.5);
    initialize_matrix(B, n, 2.0);

    /* Warm-up run */
    for (int w = 0; w < WARMUP_RUNS; w++) {
        initialize_matrix(C, n, 0.0);
        matrix_multiply_seq(A, B, C, n);
    }

    /* Measured runs */
    PerfMetrics metrics;
    double sum_time = 0.0;
    metrics.min_time = 1e9;

    for (int r = 0; r < MEASURED_RUNS; r++) {
        initialize_matrix(C, n, 0.0);
        double t_start = omp_get_wtime();
        matrix_multiply_seq(A, B, C, n);
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

    free(A);
    free(B);
    free(C);

    return metrics;
}

/* Run timing benchmark for OpenMP parallel algorithm */
PerfMetrics benchmark_openmp(int n, int threads, double seq_time) {
    double *A = allocate_matrix(n);
    double *B = allocate_matrix(n);
    double *C = allocate_matrix(n);

    initialize_matrix(A, n, 1.5);
    initialize_matrix(B, n, 2.0);

    /* Warm-up run */
    for (int w = 0; w < WARMUP_RUNS; w++) {
        initialize_matrix(C, n, 0.0);
        matrix_multiply_omp(A, B, C, n, threads);
    }

    /* Measured runs */
    PerfMetrics metrics;
    double sum_time = 0.0;
    metrics.min_time = 1e9;

    for (int r = 0; r < MEASURED_RUNS; r++) {
        initialize_matrix(C, n, 0.0);
        double t_start = omp_get_wtime();
        matrix_multiply_omp(A, B, C, n, threads);
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

    free(A);
    free(B);
    free(C);

    return metrics;
}

/* Print system and test configuration banner */
void print_banner(void) {
    printf("===========================================================================\n");
    printf(" PROJECT 1: NATIVE C / OpenMP BENCHMARK & PERFORMANCE EVALUATION SUITE\n");
    printf("===========================================================================\n");
    printf(" OpenMP Max Threads Available : %d\n", omp_get_max_threads());
    printf(" Benchmark Methodology        : %d Warmup, %d Measured Runs per configuration\n", WARMUP_RUNS, MEASURED_RUNS);
    printf(" High-Precision Timer         : OpenMP omp_get_wtime() (Wall-Clock Precision)\n");
    printf(" Matrix Dimensions Tested     : 500x500, 1000x1000, 2000x2000\n");
    printf(" OpenMP Thread Counts Tested  : 2, 4, 8 Threads\n");
    printf("===========================================================================\n\n");
}

/* Display formatted ASCII tables directly to console */
void display_results(SizeBenchmarkResult results[NUM_SIZES]) {
    printf("\n===========================================================================\n");
    printf(" PROJECT 1 -- REAL BENCHMARK RESULTS (NATIVE C / OpenMP)\n");
    printf("===========================================================================\n\n");

    /* Table 1: Execution Time */
    printf("1. EXECUTION TIME TABLE (Mean Measured Wall-Clock Time in Seconds)\n");
    printf("---------------------------------------------------------------------------\n");
    printf("%-12s | %-14s | %-12s | %-12s | %-12s\n", "Matrix", "Sequential", "2 Threads", "4 Threads", "8 Threads");
    printf("---------------------------------------------------------------------------\n");

    for (int i = 0; i < NUM_SIZES; i++) {
        char dim_str[32];
        snprintf(dim_str, sizeof(dim_str), "%dx%d", results[i].size, results[i].size);
        printf("%-12s | %-14.4f | %-12.4f | %-12.4f | %-12.4f\n",
               dim_str,
               results[i].seq_metrics.mean_time,
               results[i].omp_metrics[0].mean_time,
               results[i].omp_metrics[1].mean_time,
               results[i].omp_metrics[2].mean_time);
    }
    printf("---------------------------------------------------------------------------\n\n");

    /* Table 2: Speedup & Parallel Efficiency */
    printf("2. SPEEDUP & PARALLEL EFFICIENCY TABLE\n");
    printf("---------------------------------------------------------------------------\n");
    printf("%-12s | %-8s | %-12s | %-12s | %-15s\n", "Matrix", "Threads", "Time (s)", "Speedup", "Efficiency (%)");
    printf("---------------------------------------------------------------------------\n");

    for (int i = 0; i < NUM_SIZES; i++) {
        char dim_str[32];
        snprintf(dim_str, sizeof(dim_str), "%dx%d", results[i].size, results[i].size);
        for (int t = 0; t < NUM_THREADS; t++) {
            printf("%-12s | %-8d | %-12.4f | %-10.2fx | %-14.2f%%\n",
                   dim_str,
                   THREAD_COUNTS[t],
                   results[i].omp_metrics[t].mean_time,
                   results[i].omp_metrics[t].speedup,
                   results[i].omp_metrics[t].efficiency);
        }
    }
    printf("---------------------------------------------------------------------------\n\n");
}

/* Save benchmark results directly to CSV and TXT files using C file I/O */
void save_results_to_files(SizeBenchmarkResult results[NUM_SIZES]) {
    ensure_directory_exists("benchmark/results");
    ensure_directory_exists("results");

    const char *csv_path = "benchmark/results/project1_results.csv";
    const char *txt_path = "benchmark/results/project1_results.txt";

    /* 1. Write CSV */
    FILE *f_csv = fopen(csv_path, "w");
    if (!f_csv) {
        f_csv = fopen("project1_results.csv", "w");
        if (f_csv) csv_path = "project1_results.csv";
    }

    if (f_csv) {
        fprintf(f_csv, "Matrix_Size,Threads,Mean_Time_Sec,Min_Time_Sec,Std_Dev_Sec,Speedup,Efficiency_Pct\n");
        for (int i = 0; i < NUM_SIZES; i++) {
            int n = results[i].size;
            /* Write Sequential */
            fprintf(f_csv, "%d,1,%.6f,%.6f,%.6f,1.00,100.00\n",
                    n,
                    results[i].seq_metrics.mean_time,
                    results[i].seq_metrics.min_time,
                    results[i].seq_metrics.std_dev);
            /* Write OpenMP */
            for (int t = 0; t < NUM_THREADS; t++) {
                fprintf(f_csv, "%d,%d,%.6f,%.6f,%.6f,%.2f,%.2f\n",
                        n,
                        THREAD_COUNTS[t],
                        results[i].omp_metrics[t].mean_time,
                        results[i].omp_metrics[t].min_time,
                        results[i].omp_metrics[t].std_dev,
                        results[i].omp_metrics[t].speedup,
                        results[i].omp_metrics[t].efficiency);
            }
        }
        fclose(f_csv);
    }

    /* 2. Write TXT Report */
    FILE *f_txt = fopen(txt_path, "w");
    if (!f_txt) {
        f_txt = fopen("project1_results.txt", "w");
        if (f_txt) txt_path = "project1_results.txt";
    }

    if (f_txt) {
        time_t now = time(NULL);
        fprintf(f_txt, "============================================================\n");
        fprintf(f_txt, " PROJECT 1 -- REAL BENCHMARK RESULTS (C / OpenMP)\n");
        fprintf(f_txt, "============================================================\n");
        fprintf(f_txt, "OpenMP Max Threads Available : %d\n", omp_get_max_threads());
        fprintf(f_txt, "Benchmark Date               : %s", ctime(&now));
        fprintf(f_txt, "Warmup Runs                  : %d\n", WARMUP_RUNS);
        fprintf(f_txt, "Measured Repetitions         : %d\n", MEASURED_RUNS);
        fprintf(f_txt, "============================================================\n\n");

        fprintf(f_txt, "1. EXECUTION TIME TABLE (Seconds)\n");
        fprintf(f_txt, "---------------------------------------------------------------------------\n");
        fprintf(f_txt, "%-12s | %-14s | %-12s | %-12s | %-12s\n", "Matrix", "Sequential", "2 Threads", "4 Threads", "8 Threads");
        fprintf(f_txt, "---------------------------------------------------------------------------\n");
        for (int i = 0; i < NUM_SIZES; i++) {
            char dim_str[32];
            snprintf(dim_str, sizeof(dim_str), "%dx%d", results[i].size, results[i].size);
            fprintf(f_txt, "%-12s | %-14.4f | %-12.4f | %-12.4f | %-12.4f\n",
                    dim_str,
                    results[i].seq_metrics.mean_time,
                    results[i].omp_metrics[0].mean_time,
                    results[i].omp_metrics[1].mean_time,
                    results[i].omp_metrics[2].mean_time);
        }
        fprintf(f_txt, "---------------------------------------------------------------------------\n\n");

        fprintf(f_txt, "2. SPEEDUP & PARALLEL EFFICIENCY\n");
        fprintf(f_txt, "---------------------------------------------------------------------------\n");
        fprintf(f_txt, "%-12s | %-8s | %-12s | %-12s | %-15s\n", "Matrix", "Threads", "Time (s)", "Speedup", "Efficiency (%)");
        fprintf(f_txt, "---------------------------------------------------------------------------\n");
        for (int i = 0; i < NUM_SIZES; i++) {
            char dim_str[32];
            snprintf(dim_str, sizeof(dim_str), "%dx%d", results[i].size, results[i].size);
            for (int t = 0; t < NUM_THREADS; t++) {
                fprintf(f_txt, "%-12s | %-8d | %-12.4f | %-10.2fx | %-14.2f%%\n",
                        dim_str,
                        THREAD_COUNTS[t],
                        results[i].omp_metrics[t].mean_time,
                        results[i].omp_metrics[t].speedup,
                        results[i].omp_metrics[t].efficiency);
            }
        }
        fprintf(f_txt, "---------------------------------------------------------------------------\n");
        fclose(f_txt);
    }

    printf("[EXPORT] Benchmark results successfully exported:\n");
    printf("  -> CSV : %s\n", csv_path);
    printf("  -> TXT : %s\n\n", txt_path);
}

int main(void) {
    print_banner();

    /* 1. Run Pre-flight Validation */
    run_preflight_validation();

    /* 2. Execute Benchmark Matrix */
    printf("===========================================================================\n");
    printf(" EXECUTING NATIVE C / OpenMP BENCHMARK MATRIX\n");
    printf("===========================================================================\n");

    SizeBenchmarkResult results[NUM_SIZES];

    for (int i = 0; i < NUM_SIZES; i++) {
        int n = MATRIX_SIZES[i];
        results[i].size = n;

        printf("\n[BENCHMARK] Matrix Dimension %d x %d ...\n", n, n);

        /* Benchmark Sequential Baseline */
        printf("  -> Running Sequential baseline (%d runs)... ", MEASURED_RUNS);
        fflush(stdout);
        results[i].seq_metrics = benchmark_sequential(n);
        printf("Done. (Mean: %.4fs, Min: %.4fs)\n",
               results[i].seq_metrics.mean_time,
               results[i].seq_metrics.min_time);

        /* Benchmark OpenMP Threads */
        for (int t = 0; t < NUM_THREADS; t++) {
            int threads = THREAD_COUNTS[t];
            printf("  -> Running OpenMP (%d threads, %d runs)... ", threads, MEASURED_RUNS);
            fflush(stdout);
            results[i].omp_metrics[t] = benchmark_openmp(n, threads, results[i].seq_metrics.mean_time);
            printf("Done. (Mean: %.4fs | Speedup: %.2fx | Eff: %.1f%%)\n",
                   results[i].omp_metrics[t].mean_time,
                   results[i].omp_metrics[t].speedup,
                   results[i].omp_metrics[t].efficiency);
        }
    }

    /* 3. Render ASCII Tables */
    display_results(results);

    /* 4. Export CSV and TXT files */
    save_results_to_files(results);

    printf("===========================================================================\n");
    printf(" [PROJECT 1 BENCHMARK SUITE COMPLETED SUCCESSFULLY IN PURE C]\n");
    printf("===========================================================================\n");

    return 0;
}
