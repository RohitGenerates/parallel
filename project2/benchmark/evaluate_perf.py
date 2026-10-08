#!/usr/bin/env python3
"""
=============================================================================
Project 2: Real C/OpenMP Cache-Optimized Tiled Matrix Engine Benchmark Suite
=============================================================================
Description: Compiles and executes native C/OpenMP cache-tiled matrix
             multiplication binaries across matrix dimensions (500, 1000, 2000)
             and worker thread counts (2, 4, 8) with tile size 64x64.
             Measures real wall-clock performance, verifies numerical
             correctness against sequential baseline, calculates Speedup,
             Parallel Efficiency, and GFLOPS throughput, and exports
             structured results (CSV, TXT, JSON).
=============================================================================
"""

import os
import sys
import time
import json
import csv
import math
import shutil
import platform
import subprocess
from pathlib import Path

# ============================================================================
# CONFIGURABLE BENCHMARK PARAMETERS
# ============================================================================
WARMUP_RUNS = 1
MEASURED_RUNS = 3
MATRIX_SIZES = [500, 1000, 2000]
THREAD_COUNTS = [2, 4, 8]
TILE_SIZE = 64

# ============================================================================
# ENVIRONMENT & TOOLCHAIN DISCOVERY
# ============================================================================
def ensure_toolchain():
    """Ensure GCC is in PATH, searching standard Windows/Linux locations if needed."""
    if shutil.which("gcc"):
        return True

    candidate_paths = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.MSVCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/w64devkit/bin",
        Path("C:/msys64/mingw64/bin"),
        Path("C:/msys64/ucrt64/bin"),
        Path("C:/MinGW/bin"),
        Path("C:/Program Files/mingw-w64/x86_64-8.1.0-posix-seh-rt_v6-rev0/mingw64/bin"),
    ]

    for p in candidate_paths:
        if p.exists() and (p / "gcc.exe").exists():
            os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")
            if shutil.which("gcc"):
                return True

    return False


def get_cpu_info():
    """Retrieve detailed CPU model name across OS platforms."""
    system = platform.system()
    try:
        if system == "Windows":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu_name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            winreg.CloseKey(key)
            return cpu_name.strip()
        elif system == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
        elif system == "Darwin":
            res = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
            if res.returncode == 0:
                return res.stdout.strip()
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


def get_gcc_version():
    """Retrieve GCC version string."""
    try:
        res = subprocess.run(["gcc", "--version"], capture_output=True, text=True, check=True)
        return res.stdout.splitlines()[0].strip()
    except Exception as e:
        return f"Unavailable ({e})"


def check_openmp_support():
    """Test if GCC can compile with -fopenmp."""
    test_code = "#include <omp.h>\nint main() { return omp_get_max_threads() > 0 ? 0 : 1; }"
    try:
        res = subprocess.run(
            ["gcc", "-fopenmp", "-x", "c", "-", "-o", "omp_test_bin"],
            input=test_code,
            capture_output=True,
            text=True
        )
        if res.returncode == 0:
            exe_name = "omp_test_bin.exe" if os.name == "nt" else "omp_test_bin"
            if os.path.exists(exe_name):
                os.remove(exe_name)
            return "Available (libgomp / OpenMP supported)"
        else:
            return f"Not Supported ({res.stderr.strip()})"
    except Exception as e:
        return f"Check Failed ({e})"


def print_system_banner():
    """Display machine specs and fairness notice."""
    cpu_model = get_cpu_info()
    logical_cpus = os.cpu_count() or 1
    gcc_ver = get_gcc_version()
    openmp_status = check_openmp_support()

    print("=" * 78)
    print(" SYSTEM INFORMATION & HARDWARE TELEMETRY")
    print("=" * 78)
    print(f" CPU Model        : {cpu_model}")
    print(f" Logical Cores    : {logical_cpus}")
    print(f" Operating System : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f" GCC Version      : {gcc_ver}")
    print(f" OpenMP Support   : {openmp_status}")
    print(f" Python Version   : {platform.python_version()} ({sys.executable})")
    print(f" Architecture     : Loop Interchange (i-k-j) + L1/L2 Cache Tiling ({TILE_SIZE}x{TILE_SIZE}) + SIMD")
    print("=" * 78)
    print(" BENCHMARK FAIRNESS NOTICE:")
    print("  * Ensure your machine is connected to AC power.")
    print("  * Close background CPU-intensive applications.")
    print("  * Allow CPU thermals to stabilize before and during benchmark runs.")
    print("=" * 78)
    print()


# ============================================================================
# COMPILATION ENGINE
# ============================================================================
def compile_project2(project_dir: Path):
    """Compile Project 2 C executable with GCC -O3 and OpenMP."""
    src_dir = project_dir / "src"
    engine_src = src_dir / "matrix_engine.c"
    main_src = src_dir / "main.c"
    engine_exe = src_dir / ("matmul_engine.exe" if os.name == "nt" else "matmul_engine")

    print("[BUILD] Compiling Project 2 Matrix Engine with GCC -O3 and OpenMP...")
    cmd = ["gcc", "-O3", "-fopenmp", str(engine_src), str(main_src), "-o", str(engine_exe), "-lm"]
    print(f"  -> {' '.join(cmd)}")

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[ERROR] Matrix Engine compilation failed:\n{res.stderr}")
        sys.exit(1)

    print("[BUILD] Compilation successful.\n")
    return engine_exe


# ============================================================================
# PRE-FLIGHT VALIDATION
# ============================================================================
def run_preflight_validation(engine_exe: Path):
    """Run correctness verification check (N=500, threads=4, block=64, -v)."""
    print(f"[VALIDATION] Running pre-flight correctness check (N=500, Threads=4, Tile={TILE_SIZE})...")

    cmd = [str(engine_exe), "-n", "500", "-t", "4", "-b", str(TILE_SIZE), "-v"]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "4"

    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        print(f"[ERROR] Validation execution failed:\n{res.stderr}")
        sys.exit(1)

    if "[PASSED]" in res.stdout:
        print("  -> Correctness Check : [PASSED] Exact Numerical Equivalence")
        print("  -> Status            : ALL PRE-FLIGHT ASSERTIONS PASSED\n")
    else:
        print(f"[ERROR] Pre-flight verification failed! Output:\n{res.stdout}")
        sys.exit(1)


# ============================================================================
# BENCHMARK EXECUTION ENGINE
# ============================================================================
def execute_kernel(engine_exe: Path, n: int, kernel: str, threads: int = 1, block_size: int = TILE_SIZE):
    """
    Execute compiled engine for specified kernel, matrix size, and thread count.
    Measures wall-clock time with time.perf_counter().
    """
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)

    cmd = [
        str(engine_exe),
        "-n", str(n),
        "-t", str(threads),
        "-b", str(block_size),
        "-k", kernel
    ]

    start_time = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.perf_counter() - start_time

    if res.returncode != 0:
        raise RuntimeError(f"Execution failed for command {' '.join(cmd)}:\n{res.stderr}")

    return elapsed, res.stdout


def benchmark_configuration(engine_exe: Path, n: int, kernel: str, threads: int = 1, warmup=WARMUP_RUNS, measured=MEASURED_RUNS):
    """Run warmup and measured repetitions, returning mean, min, std-dev."""
    for _ in range(warmup):
        execute_kernel(engine_exe, n, kernel, threads)

    timings = []
    for _ in range(measured):
        t, _ = execute_kernel(engine_exe, n, kernel, threads)
        timings.append(t)

    mean_t = sum(timings) / len(timings)
    min_t = min(timings)
    std_t = math.sqrt(sum((x - mean_t) ** 2 for x in timings) / len(timings)) if len(timings) > 1 else 0.0

    return {
        "mean": mean_t,
        "min": min_t,
        "std": std_t,
        "runs": timings
    }


def run_full_benchmarks(engine_exe: Path):
    """Run full benchmark matrix across dimensions and thread configurations."""
    print("=" * 78)
    print(" EXECUTING REAL C / OpenMP CACHE-TILED BENCHMARKS")
    print(f" Configurations: Sizes={MATRIX_SIZES}, Threads={THREAD_COUNTS}, Tile Size={TILE_SIZE}x{TILE_SIZE}")
    print(f" Repetitions   : {WARMUP_RUNS} Warmup, {MEASURED_RUNS} Measured runs per test")
    print("=" * 78)

    benchmark_data = []

    for n in MATRIX_SIZES:
        print(f"\n[BENCHMARK] Dimension {n} x {n} ...")

        # 1. Sequential Baseline
        print(f"  -> Measuring Sequential baseline ({MEASURED_RUNS} runs)...", end="", flush=True)
        seq_stats = benchmark_configuration(engine_exe, n, kernel="seq", threads=1)
        t_seq = seq_stats["mean"]
        seq_flops = 2.0 * (n ** 3)
        seq_gflops = (seq_flops / t_seq) / 1e9 if t_seq > 0 else 0.0
        seq_stats["gflops"] = seq_gflops
        print(f" Done. (Mean: {t_seq:.4f}s | {seq_gflops:.2f} GFLOPS)")

        size_entry = {
            "size": n,
            "dim_str": f"{n}x{n}",
            "sequential": seq_stats,
            "threads": {}
        }

        # 2. Tiled OpenMP Runs
        for t in THREAD_COUNTS:
            print(f"  -> Measuring OpenMP Tiled ({t} threads, {MEASURED_RUNS} runs)...", end="", flush=True)
            omp_stats = benchmark_configuration(engine_exe, n, kernel="omp_tiled", threads=t)
            t_par = omp_stats["mean"]
            speedup = t_seq / t_par if t_par > 0 else 0.0
            efficiency = (speedup / t) * 100.0 if t > 0 else 0.0
            total_flops = 2.0 * (n ** 3)
            gflops = (total_flops / t_par) / 1e9 if t_par > 0 else 0.0

            omp_stats["speedup"] = speedup
            omp_stats["efficiency"] = efficiency
            omp_stats["gflops"] = gflops

            size_entry["threads"][t] = omp_stats
            print(f" Done. (Mean: {t_par:.4f}s | Speedup: {speedup:.2f}x | Eff: {efficiency:.1f}% | {gflops:.2f} GFLOPS)")

        benchmark_data.append(size_entry)

    return benchmark_data


# ============================================================================
# RESULTS FORMATTING & EXPORT
# ============================================================================
def display_results(benchmark_data: list):
    """Print clean ASCII tables matching updates.txt specifications."""
    print("\n" + "=" * 78)
    print(" PROJECT 2 -- REAL BENCHMARK RESULTS (CACHE-TILED OpenMP)")
    print("=" * 78)

    # 1. Execution Time Table
    print("\n1. WALL-CLOCK EXECUTION TIME (Mean Measured in Seconds)")
    print("-" * 78)
    header1 = f"{'Matrix Dimension':<18} | {'Sequential (s)':<15} | {'2 Threads (s)':<14} | {'4 Threads (s)':<14} | {'8 Threads (s)':<14}"
    print(header1)
    print("-" * 78)

    for entry in benchmark_data:
        dim = entry["dim_str"]
        t_seq = entry["sequential"]["mean"]
        t2 = entry["threads"][2]["mean"]
        t4 = entry["threads"][4]["mean"]
        t8 = entry["threads"][8]["mean"]
        print(f"{dim:<18} | {t_seq:<15.4f} | {t2:<14.4f} | {t4:<14.4f} | {t8:<14.4f}")
    print("-" * 78)

    # 2. Speedup, Efficiency & GFLOPS Table
    print("\n2. PARALLEL SPEEDUP, EFFICIENCY & THROUGHPUT (GFLOPS)")
    print("-" * 78)
    header2 = f"{'Matrix':<12} | {'Threads':<8} | {'Time (s)':<10} | {'Speedup':<10} | {'Efficiency':<12} | {'Throughput':<14}"
    print(header2)
    print("-" * 78)

    for entry in benchmark_data:
        dim = entry["dim_str"]
        for t in THREAD_COUNTS:
            item = entry["threads"][t]
            t_par = item["mean"]
            sp = item["speedup"]
            eff = item["efficiency"]
            gf = item["gflops"]
            print(f"{dim:<12} | {t:<8} | {t_par:<10.4f} | {sp:<10.2f}x | {eff:<11.2f}% | {gf:<10.2f} GFLOPS")
    print("-" * 78)
    print()


def save_results(benchmark_data: list, project_dir: Path):
    """Export benchmark results to CSV, TXT, and JSON files."""
    results_dir = project_dir / "benchmark" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "project2_results.csv"
    txt_path = results_dir / "project2_results.txt"
    json_path = results_dir / "project2_results.json"

    # 1. Save CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Matrix_Size", "Kernel", "Threads", "Tile_Size", "Mean_Time_Sec", "Min_Time_Sec", "Std_Dev_Sec", "Speedup", "Efficiency_Pct", "GFLOPS"])
        for entry in benchmark_data:
            n = entry["size"]
            # Write sequential
            seq = entry["sequential"]
            writer.writerow([n, "Sequential", 1, 0, f"{seq['mean']:.6f}", f"{seq['min']:.6f}", f"{seq['std']:.6f}", "1.00", "100.00", f"{seq['gflops']:.3f}"])
            # Write OpenMP Tiled
            for t in THREAD_COUNTS:
                item = entry["threads"][t]
                writer.writerow([n, "OpenMP_Tiled", t, TILE_SIZE, f"{item['mean']:.6f}", f"{item['min']:.6f}", f"{item['std']:.6f}", f"{item['speedup']:.2f}", f"{item['efficiency']:.2f}", f"{item['gflops']:.3f}"])

    # 2. Save TXT
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("============================================================\n")
        f.write(" PROJECT 2 -- REAL BENCHMARK RESULTS (Cache-Tiled OpenMP)\n")
        f.write("============================================================\n")
        f.write(f"CPU: {get_cpu_info()}\n")
        f.write(f"Logical Cores: {os.cpu_count()}\n")
        f.write(f"GCC Version: {get_gcc_version()}\n")
        f.write(f"OS: {platform.system()} {platform.release()}\n")
        f.write(f"Optimization: Loop Interchange (i-k-j) + L1/L2 Cache Tiling ({TILE_SIZE}x{TILE_SIZE}) + SIMD\n")
        f.write(f"Benchmark Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("============================================================\n\n")

        f.write("1. WALL-CLOCK EXECUTION TIME (Seconds)\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write(f"{'Matrix Dimension':<18} | {'Sequential (s)':<15} | {'2 Threads (s)':<14} | {'4 Threads (s)':<14} | {'8 Threads (s)':<14}\n")
        f.write("--------------------------------------------------------------------------------\n")
        for entry in benchmark_data:
            dim = entry["dim_str"]
            t_seq = entry["sequential"]["mean"]
            t2 = entry["threads"][2]["mean"]
            t4 = entry["threads"][4]["mean"]
            t8 = entry["threads"][8]["mean"]
            f.write(f"{dim:<18} | {t_seq:<15.4f} | {t2:<14.4f} | {t4:<14.4f} | {t8:<14.4f}\n")
        f.write("--------------------------------------------------------------------------------\n\n")

        f.write("2. PARALLEL SPEEDUP, EFFICIENCY & THROUGHPUT (GFLOPS)\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write(f"{'Matrix':<12} | {'Threads':<8} | {'Time (s)':<10} | {'Speedup':<10} | {'Efficiency':<12} | {'Throughput':<14}\n")
        f.write("--------------------------------------------------------------------------------\n")
        for entry in benchmark_data:
            dim = entry["dim_str"]
            for t in THREAD_COUNTS:
                item = entry["threads"][t]
                t_par = item["mean"]
                sp = item["speedup"]
                eff = item["efficiency"]
                gf = item["gflops"]
                f.write(f"{dim:<12} | {t:<8} | {t_par:<10.4f} | {sp:<10.2f}x | {eff:<11.2f}% | {gf:<10.2f} GFLOPS\n")
        f.write("--------------------------------------------------------------------------------\n")

    # 3. Save JSON
    json_data = {
        "metadata": {
            "project": "Project 2: Cache-Optimized Tiled Matrix Engine",
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "cpu": get_cpu_info(),
            "logical_cpus": os.cpu_count(),
            "gcc_version": get_gcc_version(),
            "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
            "tile_size": TILE_SIZE,
            "warmup_runs": WARMUP_RUNS,
            "measured_runs": MEASURED_RUNS
        },
        "results": benchmark_data
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    print(f"[EXPORT] Results successfully saved to:")
    print(f"  -> CSV  : {csv_path}")
    print(f"  -> TXT  : {txt_path}")
    print(f"  -> JSON : {json_path}\n")


# ============================================================================
# MAIN ENTRYPOINT
# ============================================================================
def main():
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent

    if not ensure_toolchain():
        print("[ERROR] GCC compiler not found in PATH or standard MinGW directories.")
        print("Please install GCC with OpenMP support (e.g., 'sudo apt install build-essential' or WinLibs MinGW).")
        sys.exit(1)

    print_system_banner()

    # Build
    engine_exe = compile_project2(project_dir)

    # Validate
    run_preflight_validation(engine_exe)

    # Benchmark
    benchmark_data = run_full_benchmarks(engine_exe)

    # Display & Export
    display_results(benchmark_data)
    save_results(benchmark_data, project_dir)

    print("=" * 78)
    print(" [PROJECT 2 BENCHMARK COMPLETED SUCCESSFULLY]")
    print("=" * 78)


if __name__ == "__main__":
    main()
