/**
 * ============================================================================
 * Project 1: OpenMP Parallel Matrix Multiplication
 * ============================================================================
 * File: matmul_omp.c
 * Description: OpenMP shared-memory parallel implementation of Matrix Multiplication.
 *              Uses row-wise static thread partitioning and thread-private accumulation
 *              registers to eliminate false sharing.
 *
 * Compilation:
 *   gcc -O3 -fopenmp matmul_omp.c -o matmul_omp -lm
 *
 * Usage:
 *   ./matmul_omp <matrix_dimension> <num_threads>
 *   Example: ./matmul_omp 1000 4
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <math.h>

/* Allocate 1D flattened matrix of size N x N */
double* allocate_matrix(int n) {
    double *mat = (double*)malloc((size_t)n * n * sizeof(double));
    if (mat == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for size %d x %d\n", n, n);
        exit(EXIT_FAILURE);
    }
    return mat;
}

/* Initialize matrix with deterministic values */
void initialize_matrix(double *mat, int n, double val) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n * n; i++) {
        mat[i] = val;
    }
}

/* OpenMP Parallel Matrix Multiplication: C = A * B */
void matrix_multiply_omp(const double *A, const double *B, double *C, int n, int num_threads) {
    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0; // Thread-private accumulation register (avoids false sharing)
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

int main(int argc, char *argv[]) {
    int n = 1000;          // Default matrix dimension
    int num_threads = 4;   // Default thread count

    if (argc > 1) {
        n = atoi(argv[1]);
        if (n <= 0) n = 1000;
    }
    if (argc > 2) {
        num_threads = atoi(argv[2]);
        if (num_threads <= 0) num_threads = 4;
    }

    printf("=========================================================\n");
    printf(" Project 1: OpenMP Parallel Matrix Multiplication\n");
    printf("=========================================================\n");
    printf("Matrix Dimension (N x N) : %d x %d\n", n, n);
    printf("Threads Configured       : %d\n", num_threads);
    printf("Max OpenMP Threads Avail : %d\n", omp_get_max_threads());
    printf("Total Memory Required    : %.2f MB\n", (3.0 * n * n * sizeof(double)) / (1024.0 * 1024.0));

    double *A = allocate_matrix(n);
    double *B = allocate_matrix(n);
    double *C = allocate_matrix(n);

    initialize_matrix(A, n, 1.5);
    initialize_matrix(B, n, 2.0);
    initialize_matrix(C, n, 0.0);

    double start_time = omp_get_wtime();
    matrix_multiply_omp(A, B, C, n, num_threads);
    double end_time = omp_get_wtime();

    double elapsed_sec = end_time - start_time;
    double expected_val = 1.5 * 2.0 * (double)n;
    double checksum = compute_checksum(C, n);

    printf("OpenMP Execution Time    : %.6f seconds\n", elapsed_sec);
    printf("Checksum                 : %.6e\n", checksum);
    printf("Verification Sample C[0] : %.2f (Expected: %.2f)\n", C[0], expected_val);

    // Check correctness
    int correct = 1;
    for (int idx = 0; idx < n * n; idx += (n + 1)) { // Check diagonal elements
        if (fabs(C[idx] - expected_val) > 1e-5) {
            correct = 0;
            break;
        }
    }

    if (correct) {
        printf("Verification Status      : [PASSED] Results match expected matrix values.\n");
    } else {
        printf("Verification Status      : [FAILED] Mismatch detected.\n");
    }
    printf("=========================================================\n");

    free(A);
    free(B);
    free(C);

    return correct ? 0 : 1;
}
