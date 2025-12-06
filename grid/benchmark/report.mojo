"""Benchmark reporting utilities for console and file output."""

from python import Python, PythonObject
from grid.benchmark.stats import BenchmarkStats
from grid.benchmark.result import BenchmarkResult


# ─────────────────────────────────────────────────────────────────────────────
# Console Output - Simple (for interactive viewer)
# ─────────────────────────────────────────────────────────────────────────────


fn print_benchmark_summary(result: BenchmarkResult) raises:
    """Print formatted benchmark summary to console (simple format)."""
    print()
    print("=" * 60)
    print("Benchmark Summary")
    print("=" * 60)
    print("Grid:", result.grid_width, "x", result.grid_height, "(", result.total_cells, "cells)")
    print()
    
    if result.has_gpu_results():
        _print_stats_block("GPU Generation", result.gpu_stats)
    
    if result.has_cpu_results():
        _print_stats_block("CPU Generation", result.cpu_stats)
    
    if result.frame_stats.count() > 0:
        _print_frame_stats(result.frame_stats, result.avg_fps())


# ─────────────────────────────────────────────────────────────────────────────
# Console Output - Detailed (for benchmark runner)
# ─────────────────────────────────────────────────────────────────────────────


fn print_detailed_report(result: BenchmarkResult) raises:
    """Print detailed benchmark report with throughput and comparison."""
    print()
    print("=" * 70)
    print("Grid Game of Life - BENCHMARK REPORT")
    print("=" * 70)
    print()
    
    _print_configuration(result)
    _print_cpu_detailed(result)
    
    if result.has_gpu_results():
        _print_gpu_detailed(result)
        _print_comparison(result)
    else:
        print("  (GPU benchmark not available - no GPU detected)")
        print()
    
    print("=" * 70)


fn _print_configuration(result: BenchmarkResult) raises:
    """Print benchmark configuration."""
    print("Configuration:")
    print("  Grid size:", result.grid_width, "x", result.grid_height)
    print("  Total cells:", result.total_cells)
    if result.num_generations > 0:
        print("  Generations:", result.num_generations)
    if result.initial_density > 0:
        print("  Initial density:", Int(result.initial_density * 100), "%")
    print()


fn _print_cpu_detailed(result: BenchmarkResult) raises:
    """Print detailed CPU stats with throughput."""
    print("-" * 70)
    print("CPU Results (SIMD + Parallel Rows):")
    print("-" * 70)
    print("  Samples:", result.cpu_stats.count())
    print("  Average:", result.cpu_stats.avg(), "ms/generation")
    print("  Minimum:", result.cpu_stats.min_val(), "ms/generation")
    print("  Maximum:", result.cpu_stats.max_val(), "ms/generation")
    print("  Std Dev:", result.cpu_stats.std_dev(), "ms")
    print("  Total time:", result.cpu_total_time(), "seconds")
    print("  Throughput:", result.cpu_generations_per_sec(), "generations/second")
    print()


fn _print_gpu_detailed(result: BenchmarkResult) raises:
    """Print detailed GPU stats with throughput."""
    print("-" * 70)
    print("GPU Results (Native GPU Kernel):")
    print("-" * 70)
    print("  Samples:", result.gpu_stats.count())
    print("  Average:", result.gpu_stats.avg(), "ms/generation")
    print("  Minimum:", result.gpu_stats.min_val(), "ms/generation")
    print("  Maximum:", result.gpu_stats.max_val(), "ms/generation")
    print("  Std Dev:", result.gpu_stats.std_dev(), "ms")
    print("  Total time:", result.gpu_total_time(), "seconds")
    print("  Throughput:", result.gpu_generations_per_sec(), "generations/second")
    print()


fn _print_comparison(result: BenchmarkResult) raises:
    """Print GPU vs CPU comparison."""
    if not result.has_comparison():
        return
    
    print("-" * 70)
    print("Comparison:")
    print("-" * 70)
    
    var speedup = result.gpu_speedup()
    if result.is_gpu_faster():
        print("  GPU is", speedup, "x faster than CPU")
    else:
        print("  CPU is", 1.0 / speedup, "x faster than GPU")
    
    print("  CPU throughput:", result.cpu_cells_per_sec() / 1_000_000.0, "M cells/second")
    print("  GPU throughput:", result.gpu_cells_per_sec() / 1_000_000.0, "M cells/second")
    print()


fn _print_stats_block(name: String, stats: BenchmarkStats) raises:
    """Print a stats block with consistent formatting."""
    print(name, "(", stats.count(), "samples):")
    print("  avg:", _format_ms(stats.avg()))
    print("  min:", _format_ms(stats.min_val()))
    print("  max:", _format_ms(stats.max_val()))
    print("  std:", _format_ms(stats.std_dev()))
    print()


fn _print_frame_stats(stats: BenchmarkStats, avg_fps: Float64) raises:
    """Print frame time stats with FPS calculation."""
    print("Frame Time (", stats.count(), "samples):")
    print("  avg:", _format_ms(stats.avg()), "(", Int(avg_fps), "FPS)")
    print("  min:", _format_ms(stats.min_val()))
    print("  max:", _format_ms(stats.max_val()))
    print()


fn _format_ms(value: Float64) raises -> String:
    """Format milliseconds value with 3 decimal places."""
    var py_builtins = Python.import_module("builtins")
    return String(py_builtins.format(value, ".3f")) + " ms"


# ─────────────────────────────────────────────────────────────────────────────
# File Output - Simple (for interactive viewer)
# ─────────────────────────────────────────────────────────────────────────────


fn save_benchmark_results(result: BenchmarkResult, filename: String) raises:
    """Save benchmark results to a text file (simple format)."""
    var datetime = Python.import_module("datetime")
    var timestamp = String(datetime.datetime.now().isoformat())
    
    var lines = List[String]()
    
    _append_simple_header(lines, timestamp, result)
    _append_simple_gpu_stats(lines, result.gpu_stats)
    _append_simple_cpu_stats(lines, result.cpu_stats)
    _append_simple_frame_stats(lines, result)
    _append_simple_comparison(lines, result)
    
    lines.append("=" * 60)
    
    _write_lines_to_file(filename, lines)
    print("Benchmark results saved to:", filename)


# ─────────────────────────────────────────────────────────────────────────────
# File Output - Detailed (for benchmark runner)
# ─────────────────────────────────────────────────────────────────────────────


fn save_detailed_report(result: BenchmarkResult, filename: String) raises:
    """Save detailed benchmark report to a text file."""
    var datetime = Python.import_module("datetime")
    var timestamp = String(datetime.datetime.now().isoformat())
    
    var lines = List[String]()
    
    _append_detailed_header(lines, timestamp, result)
    _append_detailed_stats(lines, "CPU Results (SIMD + Parallel Rows)", result.cpu_stats, result.cpu_total_time(), result.cpu_generations_per_sec())
    
    if result.has_gpu_results():
        _append_detailed_stats(lines, "GPU Results (Native GPU Kernel)", result.gpu_stats, result.gpu_total_time(), result.gpu_generations_per_sec())
        _append_detailed_comparison(lines, result)
    else:
        lines.append("  (GPU benchmark not available - no GPU detected)")
        lines.append("")
    
    lines.append("=" * 70)
    
    _write_lines_to_file(filename, lines)
    print("Report saved to:", filename)


# ─────────────────────────────────────────────────────────────────────────────
# File Output Helpers - Simple
# ─────────────────────────────────────────────────────────────────────────────


fn _append_simple_header(mut lines: List[String], timestamp: String, result: BenchmarkResult):
    """Append simple header."""
    lines.append("=" * 60)
    lines.append("Grid Game of Life - Benchmark Results")
    lines.append("=" * 60)
    lines.append("Timestamp: " + timestamp)
    lines.append("Grid: " + String(result.grid_width) + " x " + String(result.grid_height))
    lines.append("Total cells: " + String(result.total_cells))
    lines.append("")


fn _append_simple_gpu_stats(mut lines: List[String], stats: BenchmarkStats):
    """Append GPU stats if available."""
    if stats.count() > 0:
        lines.append("GPU Generation (" + String(stats.count()) + " samples):")
        lines.append("  avg: " + String(stats.avg()) + " ms")
        lines.append("  min: " + String(stats.min_val()) + " ms")
        lines.append("  max: " + String(stats.max_val()) + " ms")
        lines.append("  std: " + String(stats.std_dev()) + " ms")
        lines.append("")


fn _append_simple_cpu_stats(mut lines: List[String], stats: BenchmarkStats):
    """Append CPU stats if available."""
    if stats.count() > 0:
        lines.append("CPU Generation (" + String(stats.count()) + " samples):")
        lines.append("  avg: " + String(stats.avg()) + " ms")
        lines.append("  min: " + String(stats.min_val()) + " ms")
        lines.append("  max: " + String(stats.max_val()) + " ms")
        lines.append("  std: " + String(stats.std_dev()) + " ms")
        lines.append("")


fn _append_simple_frame_stats(mut lines: List[String], result: BenchmarkResult):
    """Append frame stats if available."""
    if result.frame_stats.count() > 0:
        lines.append("Frame Time (" + String(result.frame_stats.count()) + " samples):")
        lines.append("  avg: " + String(result.frame_stats.avg()) + " ms (" + String(Int(result.avg_fps())) + " FPS)")
        lines.append("  min: " + String(result.frame_stats.min_val()) + " ms")
        lines.append("  max: " + String(result.frame_stats.max_val()) + " ms")
        lines.append("  std: " + String(result.frame_stats.std_dev()) + " ms")
        lines.append("")


fn _append_simple_comparison(mut lines: List[String], result: BenchmarkResult):
    """Append comparison if both results available."""
    if result.has_comparison():
        var speedup = result.gpu_speedup()
        lines.append("GPU vs CPU speedup: " + String(speedup) + "x")
        lines.append("")


# ─────────────────────────────────────────────────────────────────────────────
# File Output Helpers - Detailed
# ─────────────────────────────────────────────────────────────────────────────


fn _append_detailed_header(mut lines: List[String], timestamp: String, result: BenchmarkResult):
    """Append detailed header with configuration."""
    lines.append("=" * 70)
    lines.append("Grid Game of Life - BENCHMARK REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Timestamp: " + timestamp)
    lines.append("")
    lines.append("Configuration:")
    lines.append("  Grid size: " + String(result.grid_width) + " x " + String(result.grid_height))
    lines.append("  Total cells: " + String(result.total_cells))
    if result.num_generations > 0:
        lines.append("  Generations: " + String(result.num_generations))
    if result.initial_density > 0:
        lines.append("  Initial density: " + String(Int(result.initial_density * 100)) + "%")
    lines.append("")


fn _append_detailed_stats(
    mut lines: List[String],
    title: String,
    stats: BenchmarkStats,
    total_time: Float64,
    gens_per_sec: Float64,
):
    """Append detailed stats section with throughput."""
    lines.append("-" * 70)
    lines.append(title + ":")
    lines.append("-" * 70)
    lines.append("  Samples: " + String(stats.count()))
    lines.append("  Average: " + String(stats.avg()) + " ms/generation")
    lines.append("  Minimum: " + String(stats.min_val()) + " ms/generation")
    lines.append("  Maximum: " + String(stats.max_val()) + " ms/generation")
    lines.append("  Std Dev: " + String(stats.std_dev()) + " ms")
    lines.append("  Total time: " + String(total_time) + " seconds")
    lines.append("  Throughput: " + String(gens_per_sec) + " generations/second")
    lines.append("")


fn _append_detailed_comparison(mut lines: List[String], result: BenchmarkResult):
    """Append detailed comparison section."""
    lines.append("-" * 70)
    lines.append("Comparison:")
    lines.append("-" * 70)
    
    var speedup = result.gpu_speedup()
    if result.is_gpu_faster():
        lines.append("  GPU is " + String(speedup) + "x faster than CPU")
    else:
        lines.append("  CPU is " + String(1.0 / speedup) + "x faster than GPU")
    
    lines.append("  CPU throughput: " + String(result.cpu_cells_per_sec() / 1_000_000.0) + " M cells/second")
    lines.append("  GPU throughput: " + String(result.gpu_cells_per_sec() / 1_000_000.0) + " M cells/second")
    lines.append("")


fn _write_lines_to_file(filename: String, lines: List[String]) raises:
    """Write lines to a file."""
    var pathlib = Python.import_module("pathlib")
    var path = pathlib.Path(filename)
    
    var content = String("")
    for i in range(len(lines)):
        content += lines[i] + "\n"
    
    _ = path.write_text(content)
