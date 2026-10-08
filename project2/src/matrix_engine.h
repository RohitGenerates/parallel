/**
 * ============================================================================
 * Project 2: High-Performance Cache-Optimized Matrix Engine
 * ============================================================================
 * File: matrix_engine.h
 * Description: Object-like Matrix2D abstraction with cache-tiling (blocking)
 *              and loop-interchange OpenMP parallel algorithms.
 * ============================================================================
 */

#ifndef MATRIX_ENGINE_H
#define MATRIX_ENGINE_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 2D Matrix Abstraction using contiguous row buffers */
typedef struct {
    size_t rows;
    size_t cols;
    double *raw_data;   /* Contiguous memory block */
    double **data;      /* Row pointer table for data[i][j] access */
} Matrix2D;

/* Lifecycle Management */
Matrix2D* Matrix2D_create(size_t rows, size_t cols);
void Matrix2D_free(Matrix2D *mat);
void Matrix2D_fill_constant(Matrix2D *mat, double value);
void Matrix2D_fill_pattern(Matrix2D *mat);

/* Computational Kernels */
void Matrix2D_multiply_sequential(const Matrix2D *A, const Matrix2D *B, Matrix2D *C);
void Matrix2D_multiply_tiled_omp(const Matrix2D *A, const Matrix2D *B, Matrix2D *C,
                                 int block_size, int num_threads);

/* Verification & Telemetry */
double Matrix2D_checksum(const Matrix2D *mat);
bool Matrix2D_verify_equality(const Matrix2D *C1, const Matrix2D *C2, double tolerance);

/* Full Automated C Benchmark Suite */
void Matrix2D_run_benchmark_suite(void);

#ifdef __cplusplus
}
#endif

#endif /* MATRIX_ENGINE_H */
