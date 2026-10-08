#!/usr/bin/env python3
"""
=============================================================================
Project 1: Real C/OpenMP Automated Benchmark & Performance Evaluation Suite
=============================================================================
Description: Compiles and executes native C/OpenMP matrix multiplication
             binaries across matrix dimensions (500, 1000, 2000) and thread
             counts (2, 4, 8). Measures real wall-clock performance, verifies
             numerical correctness, calculates Speedup and Parallel Efficiency,
             and exports structured results (CSV, TXT, JSON).
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

# ============================================================================
# ENVIRONMENT & TOOLCHAIN DISCOVERY
# ============================================================================
def ensure_toolchain():
    """Ensure GCC is in PATH, searching standard Windows/Linux locations if needed."""
    if shutil.which("gcc"):
        return True

    # Check common MinGW / GCC install paths on Windows
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

    print("=" * 75)
    print(" SYSTEM INFORMATION & HARDWARE TELEMETRY")
    print("=" * 75)
    print(f" CPU Model        : {cpu_model}")
    print(f" Logical Cores    : {logical_cpus}")
    print(f" Operating System : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f" GCC Version      : {gcc_ver}")
    print(f" OpenMP Support   : {openmp_status}")
    print(f" Python Version   : {platform.python_version()} ({sys.executable})")
    print("=" * 75)
    print(" BENCHMARK FAIRNESS NOTICE:")
    print("  * Ensure your machine is connected to AC power.")
    print("  * Close background CPU-intensive applications.")
    print("  * Allow CPU thermals to stabilize before and during benchmark runs.")
    print("=" * 75)
    print()


# ============================================================================
# COMPILATION ENGINE
# ============================================================================
def compile_project1(project_dir: Path):
    """Compile sequential and OpenMP C executables."""
    src_dir = project_dir / "src"
    seq_src = src_dir / "matmul_seq.c"
    omp_src = src_dir / "matmul_omp.c"

    seq_exe = src_dir / ("matmul_seq.exe" if os.name == "nt" else "matmul_seq")
    omp_exe = src_dir / ("matmul_omp.exe" if os.name == "nt" else "matmul_omp")

    print("[BUILD] Compiling C source programs with GCC -O3 optimization...")

    # Compile Sequential
    cmd_seq = ["gcc", "-O3", str(seq_src), "-o", str(seq_exe), "-lm"]
    print(f"  -> {' '.join(cmd_seq)}")
    res_seq = subprocess.run(cmd_seq, capture_output=True, text=True)
    if res_seq.returncode != 0:
        print(f"[ERROR] Sequential compilation failed:\n{res_seq.stderr}")
        sys.exit(1)

    # Compile OpenMP
    cmd_omp = ["gcc", "-O3", "-fopenmp", str(omp_src), "-o", str(omp_exe), "-lm"]
    print(f"  -> {' '.join(cmd_omp)}")
    res_omp = subprocess.run(cmd_omp, capture_output=True, text=True)
    if res_omp.returncode != 0:
        print(f"[ERROR] OpenMP compilation failed:\n{res_omp.stderr}")
        sys.exit(1)

    print("[BUILD] Compilation successful.\n")
    return seq_exe, omp_exe


# ============================================================================
# PRE-FLIGHT VALIDATION
# ============================================================================
def run_preflight_validation(seq_exe: Path, omp_exe: Path):
    """Run correctness validation on N=500 before running benchmarks."""
    print("[VALIDATION] Running pre-flight correctness check (N=500, Threads=4)...")

    # Run Seq
    res_seq = subprocess.run([str(seq_exe), "500"], capture_output=True, text=True)
    if res_seq.returncode != 0:
        print(f"[ERROR] Sequential verification execution failed:\n{res_seq.stderr}")
        sys.exit(1)

    # Run OMP
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "4"
    res_omp = subprocess.run([str(omp_exe), "500", "4"], capture_output=True, text=True, env=env)
    if res_omp.returncode != 0:
        print(f"[ERROR] OpenMP verification execution failed:\n{res_omp.stderr}")
        sys.exit(1)

    seq_passed = "[PASSED]" in res_seq.stdout
    omp_passed = "[PASSED]" in res_omp.stdout

    if seq_passed and omp_passed:
        print("  -> Sequential Verification : PASS")
        print("  -> OpenMP Verification     : PASS")
        print("  -> Status                  : ALL PRE-FLIGHT ASSERTIONS PASSED\n")
    else:
        print(f"[ERROR] Pre-flight verification failed! (Seq: {seq_passed}, OMP: {omp_passed})")
        sys.exit(1)


# ============================================================================
# BENCHMARK EXECUTION ENGINE
# ============================================================================
def execute_binary(exe_path: Path, args: list, num_threads: int = None):
    """
    Execute binary via subprocess, setting OMP_NUM_THREADS, and measure
    reliable wall-clock execution time with time.perf_counter().
    """
    env = os.environ.copy()
    if num_threads is not None:
        env["OMP_NUM_THREADS"] = str(num_threads)

    cmd = [str(exe_path)] + [str(a) for a in args]

    start_time = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.perf_counter() - start_time

    if res.returncode != 0:
        raise RuntimeError(f"Execution failed for command {' '.join(cmd)}:\n{res.stderr}")

    return elapsed, res.stdout


def run_single_benchmark(exe_path: Path, n: int, threads: int = None, warmup=WARMUP_RUNS, measured=MEASURED_RUNS):
    """Run warmup and measured repetitions, returning mean, min, std-dev."""
    args = [n] if threads is None else [n, threads]

    # Warm-up runs (not recorded)
    for _ in range(warmup):
        execute_binary(exe_path, args, threads)

    # Measured runs
    timings = []
    for _ in range(measured):
        t, _ = execute_binary(exe_path, args, threads)
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


def run_full_benchmarks(seq_exe: Path, omp_exe: Path):
    """Run complete benchmark suite across matrix sizes and thread counts."""
    print("=" * 75)
    print(" EXECUTING REAL C / OpenMP BENCHMARKS")
    print(f" Configurations: Sizes={MATRIX_SIZES}, Threads={THREAD_COUNTS}")
    print(f" Repetitions   : {WARMUP_RUNS} Warmup, {MEASURED_RUNS} Measured runs per test")
    print("=" * 75)

    benchmark_data = []

    for n in MATRIX_SIZES:
        print(f"\n[BENCHMARK] Dimension {n} x {n} ...")

        # 1. Sequential Run
        print(f"  -> Running Sequential baseline ({MEASURED_RUNS} runs)...", end="", flush=True)
        seq_stats = run_single_benchmark(seq_exe, n, threads=None)
        t_seq = seq_stats["mean"]
        print(f" Done. (Mean: {t_seq:.4f}s, Min: {seq_stats['min']:.4f}s)")

        size_entry = {
            "size": n,
            "dim_str": f"{n}x{n}",
            "sequential": seq_stats,
            "threads": {}
        }

        # 2. OpenMP Runs
        for t in THREAD_COUNTS:
            print(f"  -> Running OpenMP ({t} threads, {MEASURED_RUNS} runs)...", end="", flush=True)
            omp_stats = run_single_benchmark(omp_exe, n, threads=t)
            t_par = omp_stats["mean"]
            speedup = t_seq / t_par if t_par > 0 else 0.0
            efficiency = (speedup / t) * 100.0 if t > 0 else 0.0

            omp_stats["speedup"] = speedup
            omp_stats["efficiency"] = efficiency
            size_entry["threads"][t] = omp_stats

            print(f" Done. (Mean: {t_par:.4f}s | Speedup: {speedup:.2f}x | Eff: {efficiency:.1f}%)")

        benchmark_data.append(size_entry)

    return benchmark_data


# ============================================================================
# RESULTS FORMATTING & EXPORT
# ============================================================================
def display_results(benchmark_data: list):
    """Print clean ASCII tables matching updates.txt specifications."""
    print("\n" + "=" * 75)
    print(" PROJECT 1 -- REAL BENCHMARK RESULTS")
    print("=" * 75)

    # 1. Execution Time Table
    print("\n1. EXECUTION TIME TABLE (Mean Measured Wall-Clock Time in Seconds)")
    print("-" * 75)
    header1 = f"{'Matrix':<12} | {'Sequential':<14} | {'2 Threads':<12} | {'4 Threads':<12} | {'8 Threads':<12}"
    print(header1)
    print("-" * 75)

    for entry in benchmark_data:
        dim = entry["dim_str"]
        t_seq = entry["sequential"]["mean"]
        t2 = entry["threads"][2]["mean"]
        t4 = entry["threads"][4]["mean"]
        t8 = entry["threads"][8]["mean"]
        print(f"{dim:<12} | {t_seq:<14.4f} | {t2:<12.4f} | {t4:<12.4f} | {t8:<12.4f}")
    print("-" * 75)

    # 2. Speedup & Parallel Efficiency Table
    print("\n2. SPEEDUP & PARALLEL EFFICIENCY TABLE")
    print("-" * 75)
    header2 = f"{'Matrix':<12} | {'Threads':<8} | {'Time (s)':<12} | {'Speedup':<12} | {'Efficiency (%)':<15}"
    print(header2)
    print("-" * 75)

    for entry in benchmark_data:
        dim = entry["dim_str"]
        for t in THREAD_COUNTS:
            t_par = entry["threads"][t]["mean"]
            sp = entry["threads"][t]["speedup"]
            eff = entry["threads"][t]["efficiency"]
            print(f"{dim:<12} | {t:<8} | {t_par:<12.4f} | {sp:<12.2f}x | {eff:<15.2f}%")
    print("-" * 75)
    print()


def save_results(benchmark_data: list, project_dir: Path):
    """Export benchmark results to CSV, TXT, and JSON files."""
    results_dir = project_dir / "benchmark" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "project1_results.csv"
    txt_path = results_dir / "project1_results.txt"
    json_path = results_dir / "project1_results.json"

    # 1. Save CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Matrix_Size", "Threads", "Mean_Time_Sec", "Min_Time_Sec", "Std_Dev_Sec", "Speedup", "Efficiency_Pct"])
        for entry in benchmark_data:
            n = entry["size"]
            # Write sequential
            writer.writerow([n, 1, f"{entry['sequential']['mean']:.6f}", f"{entry['sequential']['min']:.6f}", f"{entry['sequential']['std']:.6f}", "1.00", "100.00"])
            # Write OpenMP
            for t in THREAD_COUNTS:
                item = entry["threads"][t]
                writer.writerow([n, t, f"{item['mean']:.6f}", f"{item['min']:.6f}", f"{item['std']:.6f}", f"{item['speedup']:.2f}", f"{item['efficiency']:.2f}"])

    # 2. Save TXT
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("============================================================\n")
        f.write(" PROJECT 1 -- REAL BENCHMARK RESULTS (C / OpenMP)\n")
        f.write("============================================================\n")
        f.write(f"CPU: {get_cpu_info()}\n")
        f.write(f"Logical Cores: {os.cpu_count()}\n")
        f.write(f"GCC Version: {get_gcc_version()}\n")
        f.write(f"OS: {platform.system()} {platform.release()}\n")
        f.write(f"Benchmark Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("============================================================\n\n")

        f.write("1. EXECUTION TIME (Seconds)\n")
        f.write("----------------------------------------------------------------------\n")
        f.write(f"{'Matrix':<12} | {'Sequential':<14} | {'2 Threads':<12} | {'4 Threads':<12} | {'8 Threads':<12}\n")
        f.write("----------------------------------------------------------------------\n")
        for entry in benchmark_data:
            dim = entry["dim_str"]
            t_seq = entry["sequential"]["mean"]
            t2 = entry["threads"][2]["mean"]
            t4 = entry["threads"][4]["mean"]
            t8 = entry["threads"][8]["mean"]
            f.write(f"{dim:<12} | {t_seq:<14.4f} | {t2:<12.4f} | {t4:<12.4f} | {t8:<12.4f}\n")
        f.write("----------------------------------------------------------------------\n\n")

        f.write("2. SPEEDUP & PARALLEL EFFICIENCY\n")
        f.write("----------------------------------------------------------------------\n")
        f.write(f"{'Matrix':<12} | {'Threads':<8} | {'Time (s)':<12} | {'Speedup':<12} | {'Efficiency (%)':<15}\n")
        f.write("----------------------------------------------------------------------\n")
        for entry in benchmark_data:
            dim = entry["dim_str"]
            for t in THREAD_COUNTS:
                t_par = entry["threads"][t]["mean"]
                sp = entry["threads"][t]["speedup"]
                eff = entry["threads"][t]["efficiency"]
                f.write(f"{dim:<12} | {t:<8} | {t_par:<12.4f} | {sp:<12.2f}x | {eff:<15.2f}%\n")
        f.write("----------------------------------------------------------------------\n")

    # 3. Save JSON
    json_data = {
        "metadata": {
            "project": "Project 1: Standard Row-Major Matrix Multiplication",
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "cpu": get_cpu_info(),
            "logical_cpus": os.cpu_count(),
            "gcc_version": get_gcc_version(),
            "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
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
    # Resolve project1 base directory
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent

    # Toolchain check
    if not ensure_toolchain():
        print("[ERROR] GCC compiler not found in PATH or standard MinGW directories.")
        print("Please install GCC with OpenMP support (e.g., 'sudo apt install build-essential' or WinLibs MinGW).")
        sys.exit(1)

    print_system_banner()

    # Build
    seq_exe, omp_exe = compile_project1(project_dir)

    # Validate
    run_preflight_validation(seq_exe, omp_exe)

    # Benchmark
    benchmark_data = run_full_benchmarks(seq_exe, omp_exe)

    # Display & Export
    display_results(benchmark_data)
    save_results(benchmark_data, project_dir)

    print("=" * 75)
    print(" [PROJECT 1 BENCHMARK COMPLETED SUCCESSFULLY]")
    print("=" * 75)


if __name__ == "__main__":
    main()
