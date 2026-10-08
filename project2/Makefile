# ==============================================================================
# Project 2: Cache-Optimized Tiled Matrix Multiplication Engine
# Makefile for Native C Compilation and Benchmarking
# ==============================================================================

CC = gcc
CFLAGS = -O3 -Wall -Wextra -static
OMPFLAGS = -fopenmp
LDFLAGS = -lm

SRC_DIR = src
BIN_DIR = bin

TARGET_ENGINE = $(BIN_DIR)/matmul_engine

.PHONY: all clean run benchmark directories

all: directories $(TARGET_ENGINE)

directories:
	@mkdir -p $(BIN_DIR) 2>/dev/null || mkdir $(BIN_DIR) 2>nul || true

$(TARGET_ENGINE): $(SRC_DIR)/matrix_engine.c $(SRC_DIR)/main.c
	$(CC) $(CFLAGS) $(OMPFLAGS) $^ -o $@ $(LDFLAGS)

benchmark: $(TARGET_ENGINE)
	@$(TARGET_ENGINE) --benchmark

run: benchmark

clean:
	@rm -rf $(BIN_DIR)/*.exe $(BIN_DIR)/matmul_engine 2>/dev/null || true
