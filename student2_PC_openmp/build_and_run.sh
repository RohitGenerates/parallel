#!/bin/bash

echo "Compiling..."
# Ensure optimization is O0 or O1 to prevent MinGW GCC openmp segfaults
gcc -O0 -fopenmp source/main.c -o benchmark/pc_prime_sim.exe -lm

if [ $? -eq 0 ]; then
    echo "Compilation successful. Running benchmark..."
    cd benchmark
    ./pc_prime_sim.exe
else
    echo "Compilation failed."
fi
