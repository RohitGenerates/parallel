# ==============================================================================
# High-Performance Parallel Matrix Multiplication Suite (C & OpenMP)
# Root Makefile for Compiling and Benchmarking Project 1 & Project 2
# ==============================================================================

MAKE_CMD ?= $(MAKE)

.PHONY: all project1 project2 run run-project1 run-project2 clean

all: project1 project2

project1:
	@$(MAKE_CMD) -C project1 all

project2:
	@$(MAKE_CMD) -C project2 all

run: run-project1 run-project2

run-project1: project1
	@echo ""
	@echo "=============================================================================="
	@echo " RUNNING PROJECT 1 NATIVE C BENCHMARK SUITE"
	@echo "=============================================================================="
	@$(MAKE_CMD) -C project1 run

run-project2: project2
	@echo ""
	@echo "=============================================================================="
	@echo " RUNNING PROJECT 2 NATIVE C BENCHMARK SUITE"
	@echo "=============================================================================="
	@$(MAKE_CMD) -C project2 run

clean:
	@$(MAKE_CMD) -C project1 clean
	@$(MAKE_CMD) -C project2 clean
