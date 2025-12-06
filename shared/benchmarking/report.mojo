"""Benchmark reporting utilities for console and file output (shared)."""

from python import Python, PythonObject
from shared.benchmarking.stats import BenchmarkStats
from shared.benchmarking.result import BenchmarkResult


struct LineSink:
    """Collects lines and optionally echoes to console."""
    var lines: List[String]
    var echo: Bool
    
    fn __init__(out self, echo: Bool = False):
        self.lines = List[String]()
        self.echo = echo
    
    fn emit(mut self, line: String) raises:
        if self.echo:
            print(line)
        self.lines.append(line)
    
    fn emit_blank(mut self) raises:
        self.emit("")


# ─────────────────────────────────────────────────────────────────────────────
# Simple Summary (shared renderer for console + file)
# ─────────────────────────────────────────────────────────────────────────────


fn print_benchmark_summary(result: BenchmarkResult) raises:
    """Print formatted benchmark summary to console (simple format)."""
    var sink = LineSink(True)
    sink.emit_blank()
    _render_simple_summary(result, sink)


# ─────────────────────────────────────────────────────────────────────────────
# Console Output - Detailed (for benchmark runner)
# ─────────────────────────────────────────────────────────────────────────────


fn print_detailed_report(result: BenchmarkResult) raises:
    """Print detailed benchmark report with throughput and comparison."""
    var sink = LineSink(True)
    _render_detailed_report(result, sink)


fn _render_detailed_report(result: BenchmarkResult, mut sink: LineSink, timestamp: String = "") raises:
    """Render detailed report to a sink (echoes if enabled)."""
    sink.emit("")
    sink.emit("=" * 70)
    sink.emit("Grid Automata - BENCHMARK REPORT")
    sink.emit("=" * 70)
    sink.emit("")
    if len(timestamp) > 0:
        sink.emit("Timestamp: " + timestamp)
        sink.emit("")
    
    _emit_configuration(result, sink)
    _emit_cpu_detailed(result, sink)
    
    if result.has_gpu_results():
        _emit_gpu_detailed(result, sink)
        _emit_comparison(result, sink)
    else:
        sink.emit("  (GPU benchmark not available - no GPU detected)")
        sink.emit_blank()
    
    sink.emit("=" * 70)


fn _emit_configuration(result: BenchmarkResult, mut sink: LineSink) raises:
    """Emit benchmark configuration."""
    sink.emit("Configuration:")
    sink.emit("  Grid size: " + String(result.grid_width) + " x " + String(result.grid_height))
    sink.emit("  Total cells: " + String(result.total_cells))
    if result.num_generations > 0:
        sink.emit("  Generations: " + String(result.num_generations))
    if result.initial_density > 0:
        sink.emit("  Initial density: " + String(Int(result.initial_density * 100)) + " %")
    sink.emit_blank()


fn _emit_cpu_detailed(result: BenchmarkResult, mut sink: LineSink) raises:
    """Emit detailed CPU stats with throughput."""
    sink.emit("-" * 70)
    sink.emit("CPU Results (SIMD + Parallel Rows):")
    sink.emit("-" * 70)
    sink.emit("  Samples: " + String(result.cpu_stats.count()))
    sink.emit("  Average: " + _format_ms(result.cpu_stats.avg()) + " per generation")
    sink.emit("  Minimum: " + _format_ms(result.cpu_stats.min_val()) + " per generation")
    sink.emit("  Maximum: " + _format_ms(result.cpu_stats.max_val()) + " per generation")
    sink.emit("  Std Dev: " + _format_ms(result.cpu_stats.std_dev()))
    sink.emit("  Total time: " + _format_sec3(result.cpu_total_time()) + " seconds")
    sink.emit("  Throughput: " + _format_int(result.cpu_generations_per_sec()) + " generations/second")
    sink.emit_blank()


fn _emit_gpu_detailed(result: BenchmarkResult, mut sink: LineSink) raises:
    """Emit detailed GPU stats with throughput."""
    sink.emit("-" * 70)
    sink.emit("GPU Results (Native GPU Kernel):")
    sink.emit("-" * 70)
    sink.emit("  Samples: " + String(result.gpu_stats.count()))
    sink.emit("  Average: " + _format_ms(result.gpu_stats.avg()) + " per generation")
    sink.emit("  Minimum: " + _format_ms(result.gpu_stats.min_val()) + " per generation")
    sink.emit("  Maximum: " + _format_ms(result.gpu_stats.max_val()) + " per generation")
    sink.emit("  Std Dev: " + _format_ms(result.gpu_stats.std_dev()))
    sink.emit("  Total time: " + _format_sec3(result.gpu_total_time()) + " seconds")
    sink.emit("  Throughput: " + _format_int(result.gpu_generations_per_sec()) + " generations/second")
    sink.emit_blank()


fn _emit_comparison(result: BenchmarkResult, mut sink: LineSink) raises:
    """Emit GPU vs CPU comparison."""
    if not result.has_comparison():
        return
    
    sink.emit("-" * 70)
    sink.emit("Comparison:")
    sink.emit("-" * 70)
    
    var speedup = result.gpu_speedup()
    if result.is_gpu_faster():
        sink.emit("  GPU is " + _format_dec2(speedup) + " x faster than CPU")
    else:
        sink.emit("  CPU is " + _format_dec2(1.0 / speedup) + " x faster than GPU")
    
    sink.emit("  CPU throughput: " + _format_int(result.cpu_cells_per_sec() / 1_000_000.0) + " M cells/second")
    sink.emit("  GPU throughput: " + _format_int(result.gpu_cells_per_sec() / 1_000_000.0) + " M cells/second")
    sink.emit_blank()


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


fn _format_dec2(value: Float64) raises -> String:
    """Format float with 2 decimal places."""
    var py_builtins = Python.import_module("builtins")
    return String(py_builtins.format(value, ".2f"))


fn _format_sec3(value: Float64) raises -> String:
    """Format seconds with 3 decimal places."""
    var py_builtins = Python.import_module("builtins")
    return String(py_builtins.format(value, ".3f"))


fn _format_int(value: Float64) -> String:
    """Format float to whole-number string (truncated)."""
    return String(Int(value))


# ─────────────────────────────────────────────────────────────────────────────
# File Output - Simple (for interactive viewer)
# ─────────────────────────────────────────────────────────────────────────────


fn save_benchmark_results(result: BenchmarkResult, filename: String) raises:
    """Save benchmark results to a text file (simple format)."""
    var sink = LineSink(False)
    var datetime = Python.import_module("datetime")
    var timestamp = String(datetime.datetime.now().isoformat())
    _render_simple_summary(result, sink, timestamp)
    sink.emit("=" * 60)
    _write_lines_to_file(filename, sink.lines)
    print("Benchmark results saved to:", filename)


# ─────────────────────────────────────────────────────────────────────────────
# File Output - Detailed (for benchmark runner)
# ─────────────────────────────────────────────────────────────────────────────


fn save_detailed_report(result: BenchmarkResult, filename: String) raises:
    """Save detailed benchmark report to a text file."""
    var datetime = Python.import_module("datetime")
    var timestamp = String(datetime.datetime.now().isoformat())
    
    var sink = LineSink(False)
    _render_detailed_report(result, sink, timestamp)
    _write_lines_to_file(filename, sink.lines)
    print("Report saved to:", filename)


# ─────────────────────────────────────────────────────────────────────────────
# Simple summary rendering (shared sink)
# ─────────────────────────────────────────────────────────────────────────────


fn _render_simple_summary(result: BenchmarkResult, mut sink: LineSink, timestamp: String = "") raises:
    sink.emit("=" * 60)
    sink.emit("Grid Automata - Benchmark Results")
    sink.emit("=" * 60)
    if len(timestamp) > 0:
        sink.emit("Timestamp: " + timestamp)
    sink.emit("Grid: " + String(result.grid_width) + " x " + String(result.grid_height))
    sink.emit("Total cells: " + String(result.total_cells))
    sink.emit_blank()
    
    if result.has_gpu_results():
        _emit_simple_stats_block("GPU Generation", result.gpu_stats, sink)
    
    if result.has_cpu_results():
        _emit_simple_stats_block("CPU Generation", result.cpu_stats, sink)
    
    if result.frame_stats.count() > 0:
        _emit_simple_frame_stats(result, sink)
    
    if result.has_comparison():
        var speedup = result.gpu_speedup()
        sink.emit("GPU vs CPU speedup: " + _format_dec2(speedup) + "x")
        sink.emit_blank()


fn _emit_simple_stats_block(name: String, stats: BenchmarkStats, mut sink: LineSink) raises:
    if stats.count() > 0:
        sink.emit(name + " (" + String(stats.count()) + " samples):")
        sink.emit("  avg: " + _format_ms(stats.avg()))
        sink.emit("  min: " + _format_ms(stats.min_val()))
        sink.emit("  max: " + _format_ms(stats.max_val()))
        sink.emit("  std: " + _format_ms(stats.std_dev()))
        sink.emit_blank()


fn _emit_simple_frame_stats(result: BenchmarkResult, mut sink: LineSink) raises:
    sink.emit("Frame Time (" + String(result.frame_stats.count()) + " samples):")
    sink.emit("  avg: " + _format_ms(result.frame_stats.avg()) + " (" + String(Int(result.avg_fps())) + " FPS)")
    sink.emit("  min: " + _format_ms(result.frame_stats.min_val()))
    sink.emit("  max: " + _format_ms(result.frame_stats.max_val()))
    sink.emit("  std: " + _format_ms(result.frame_stats.std_dev()))
    sink.emit_blank()


# ─────────────────────────────────────────────────────────────────────────────
# File Output Helpers - Detailed
# ─────────────────────────────────────────────────────────────────────────────


fn _append_detailed_header(mut lines: List[String], timestamp: String, result: BenchmarkResult):
    """Append detailed header with configuration."""
    lines.append("=" * 70)
    lines.append("Grid Automata - BENCHMARK REPORT")
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
) raises:
    """Append detailed stats section with throughput."""
    lines.append("-" * 70)
    lines.append(title + ":")
    lines.append("-" * 70)
    lines.append("  Samples: " + String(stats.count()))
    lines.append("  Average: " + _format_ms(stats.avg()) + " per generation")
    lines.append("  Minimum: " + _format_ms(stats.min_val()) + " per generation")
    lines.append("  Maximum: " + _format_ms(stats.max_val()) + " per generation")
    lines.append("  Std Dev: " + _format_ms(stats.std_dev()))
    lines.append("  Total time: " + _format_sec3(total_time) + " seconds")
    lines.append("  Throughput: " + _format_int(gens_per_sec) + " generations/second")
    lines.append("")


fn _append_detailed_comparison(mut lines: List[String], result: BenchmarkResult) raises:
    """Append detailed comparison section."""
    lines.append("-" * 70)
    lines.append("Comparison:")
    lines.append("-" * 70)
    
    var speedup = result.gpu_speedup()
    if result.is_gpu_faster():
        lines.append("  GPU is " + _format_dec2(speedup) + "x faster than CPU")
    else:
        lines.append("  CPU is " + _format_dec2(1.0 / speedup) + "x faster than GPU")
    
    lines.append("  CPU throughput: " + _format_int(result.cpu_cells_per_sec() / 1_000_000.0) + " M cells/second")
    lines.append("  GPU throughput: " + _format_int(result.gpu_cells_per_sec() / 1_000_000.0) + " M cells/second")
    lines.append("")


fn _write_lines_to_file(filename: String, lines: List[String]) raises:
    """Write lines to a file."""
    var pathlib = Python.import_module("pathlib")
    var path = pathlib.Path(filename)
    
    var content = String("")
    for i in range(len(lines)):
        content += lines[i] + "\n"
    
    _ = path.write_text(content)

