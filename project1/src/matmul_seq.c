/**
 * ============================================================================
 * Project 1: Sequential Matrix Multiplication Baseline
 * ============================================================================
 * File: matmul_seq.c
 * Description: Standard sequential implementation of Matrix Multiplication
 *              using 1D-flattened dynamic heap allocation.
 *
 * Compilation:
 *   gcc -O3 matmul_seq.c -o matmul_seq -lm
 *
 * Usage:
 *   ./matmul_seq <matrix_dimension>
 *   Example: ./matmul_seq 1000
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#if defined(_WIN32)
#include <windows.h>
static double get_wall_time(void) {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart;
}
#else
static double get_wall_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}
#endif

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
    for (int i = 0; i < n * n; i++) {
        mat[i] = val;
    }
}

/* Sequential Matrix Multiplication: C = A * B */
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

/* Compute checksum of matrix */
double compute_checksum(const double *mat, int n) {
    double sum = 0.0;
    for (int i = 0; i < n * n; i++) {
        sum += fabs(mat[i]);
    }
    return sum;
}

int main(int argc, char *argv[]) {
    int n = 1000; // Default matrix dimension

    if (argc > 1) {
        n = atoi(argv[1]);
        if (n <= 0) {
            fprintf(stderr, "Invalid matrix size: %s. Using default 1000.\n", argv[1]);
            n = 1000;
        }
    }

    printf("=========================================================\n");
    printf(" Project 1: Sequential Matrix Multiplication (C Baseline)\n");
    printf("=========================================================\n");
    printf("Matrix Dimension (N x N) : %d x %d\n", n, n);
    printf("Total Memory Required    : %.2f MB\n", (3.0 * n * n * sizeof(double)) / (1024.0 * 1024.0));

    double *A = allocate_matrix(n);
    double *B = allocate_matrix(n);
    double *C = allocate_matrix(n);

    initialize_matrix(A, n, 1.5);
    initialize_matrix(B, n, 2.0);
    initialize_matrix(C, n, 0.0);

    double start_time = get_wall_time();
    matrix_multiply_seq(A, B, C, n);
    double end_time = get_wall_time();

    double elapsed_sec = end_time - start_time;
    double expected_val = 1.5 * 2.0 * (double)n;
    double checksum = compute_checksum(C, n);

    printf("Sequential Execution Time: %.6f seconds\n", elapsed_sec);
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
