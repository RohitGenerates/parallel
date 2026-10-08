# ==============================================================================
# Project 1: Standard Row-Major Parallel Matrix Multiplication
# Makefile for Native C Compilation and Benchmarking
# ==============================================================================

CC = gcc
CFLAGS = -O3 -Wall -Wextra -static
OMPFLAGS = -fopenmp
LDFLAGS = -lm

SRC_DIR = src
BIN_DIR = bin

TARGET_SEQ = $(BIN_DIR)/matmul_seq
TARGET_OMP = $(BIN_DIR)/matmul_omp
TARGET_BENCH = $(BIN_DIR)/benchmark

.PHONY: all clean run benchmark directories

all: directories $(TARGET_SEQ) $(TARGET_OMP) $(TARGET_BENCH)

directories:
	@mkdir -p $(BIN_DIR) 2>/dev/null || mkdir $(BIN_DIR) 2>nul || true

$(TARGET_SEQ): $(SRC_DIR)/matmul_seq.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

$(TARGET_OMP): $(SRC_DIR)/matmul_omp.c
	$(CC) $(CFLAGS) $(OMPFLAGS) $< -o $@ $(LDFLAGS)

$(TARGET_BENCH): $(SRC_DIR)/benchmark.c
	$(CC) $(CFLAGS) $(OMPFLAGS) $< -o $@ $(LDFLAGS)

benchmark: $(TARGET_BENCH)
	@$(TARGET_BENCH)

run: benchmark

clean:
	@rm -rf $(BIN_DIR)/*.exe $(BIN_DIR)/matmul_seq $(BIN_DIR)/matmul_omp $(BIN_DIR)/benchmark 2>/dev/null || true
