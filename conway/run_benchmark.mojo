"""Conway's Game of Life - Performance Benchmark

Runs 10,000 generations as fast as possible for both GPU and CPU modes,
then saves a comparison report to a text file.

Run: pixi run mojo conway/run_benchmark.mojo
"""

from python import Python, PythonObject
from sys import has_accelerator
from conway.grid import Grid, INITIAL_DENSITY
from conway.cpu_compute import CPUCompute
from conway.benchmark import (
    BenchmarkStats,
    BenchmarkResult,
    print_detailed_report,
    save_detailed_report,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# 4K resolution for benchmark (larger grid = better stress test)
alias BENCH_WIDTH: Int = 3840
alias BENCH_HEIGHT: Int = 2160
alias NUM_GENERATIONS: Int = 10_000
alias BENCHMARK_OUTPUT_FILE: String = "conway_benchmark.txt"


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark Runners
# ─────────────────────────────────────────────────────────────────────────────


fn _run_cpu_benchmark(mut grid: Grid, py_time: PythonObject) raises -> BenchmarkStats:
    """Run CPU benchmark for NUM_GENERATIONS generations."""
    var stats = BenchmarkStats("CPU")
    
    print("Running CPU benchmark (", NUM_GENERATIONS, "generations)...")
    
    var total_start = py_time.time()
    
    for gen in range(NUM_GENERATIONS):
        var gen_start = py_time.time()
        CPUCompute.step(grid)
        var gen_time = Float64(py_time.time() - gen_start) * 1000.0
        stats.add(gen_time)
        
        if (gen + 1) % 1000 == 0:
            print("  CPU:", gen + 1, "/", NUM_GENERATIONS, "generations")
    
    _print_benchmark_complete("CPU", py_time.time() - total_start)
    return stats^


fn _run_gpu_benchmark(mut grid: Grid, py_time: PythonObject) raises -> BenchmarkStats:
    """Run GPU benchmark for NUM_GENERATIONS generations."""
    from conway.gpu_compute import GPUCompute
    
    var stats = BenchmarkStats("GPU")
    
    print("Running GPU benchmark (", NUM_GENERATIONS, "generations)...")
    
    var gpu_compute = GPUCompute(grid.width, grid.height, grid.stride)
    var src_ptr = grid.cells_a.unsafe_ptr() if grid.active == 0 else grid.cells_b.unsafe_ptr()
    gpu_compute.upload_from_cpu(src_ptr, grid.active)
    
    var total_start = py_time.time()
    
    for gen in range(NUM_GENERATIONS):
        var gen_start = py_time.time()
        gpu_compute.step()
        var gen_time = Float64(py_time.time() - gen_start) * 1000.0
        stats.add(gen_time)
        
        if (gen + 1) % 1000 == 0:
            print("  GPU:", gen + 1, "/", NUM_GENERATIONS, "generations")
    
    _print_benchmark_complete("GPU", py_time.time() - total_start)
    return stats^


fn _print_benchmark_complete(mode: String, total_time: PythonObject) raises:
    """Print benchmark completion message."""
    var time_secs = Float64(total_time)
    var gens_per_sec = Float64(NUM_GENERATIONS) / time_secs
    print(mode, "benchmark complete:")
    print("  Total time:", time_secs, "seconds")
    print("  Generations/sec:", gens_per_sec)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────


fn run_benchmark() raises:
    """Run the complete benchmark suite."""
    var py_time = Python.import_module("time")
    
    _print_banner()
    
    var has_gpu = _check_gpu()
    var gpu_stats = _run_gpu_if_available(has_gpu, py_time)
    var cpu_stats = _run_cpu(py_time)
    
    # Create result and generate reports
    var frame_stats = BenchmarkStats("Frame")
    var result = BenchmarkResult(
        gpu_stats^, cpu_stats^, frame_stats^,
        BENCH_WIDTH, BENCH_HEIGHT,
        NUM_GENERATIONS, INITIAL_DENSITY
    )
    
    print_detailed_report(result)
    save_detailed_report(result, BENCHMARK_OUTPUT_FILE)


fn _print_banner() raises:
    """Print benchmark startup banner."""
    print("=" * 70)
    print("CONWAY'S GAME OF LIFE - PERFORMANCE BENCHMARK")
    print("=" * 70)
    print()
    print("Grid:", BENCH_WIDTH, "x", BENCH_HEIGHT, "(", BENCH_WIDTH * BENCH_HEIGHT, "cells)")
    print("Generations:", NUM_GENERATIONS)
    print()


fn _check_gpu() -> Bool:
    """Check GPU availability."""
    @parameter
    if has_accelerator():
        return True
    return False


fn _run_gpu_if_available(has_gpu: Bool, py_time: PythonObject) raises -> BenchmarkStats:
    """Run GPU benchmark if available, return empty stats otherwise."""
    if has_gpu:
        print("GPU detected, running GPU benchmark first...")
        print()
        var grid = Grid(BENCH_WIDTH, BENCH_HEIGHT)
        grid.randomize(INITIAL_DENSITY)
        return _run_gpu_benchmark(grid, py_time)
    else:
        print("No GPU detected, skipping GPU benchmark")
        print()
        return BenchmarkStats("GPU")


fn _run_cpu(py_time: PythonObject) raises -> BenchmarkStats:
    """Run CPU benchmark."""
    print("Running CPU benchmark...")
    print()
    var grid = Grid(BENCH_WIDTH, BENCH_HEIGHT)
    grid.randomize(INITIAL_DENSITY)
    return _run_cpu_benchmark(grid, py_time)


fn main() raises:
    run_benchmark()
