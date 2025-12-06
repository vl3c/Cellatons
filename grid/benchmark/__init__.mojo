# Benchmark module for Grid Game of Life
#
# Provides timing statistics collection and reporting utilities.
#
# Files:
# - stats.mojo: BenchmarkStats struct for collecting timing samples
# - result.mojo: BenchmarkResult struct for complete benchmark data
# - report.mojo: Functions for printing and saving reports

from shared.benchmarking import (
    BenchmarkStats,
    BenchmarkResult,
    print_benchmark_summary,
    print_detailed_report,
    save_benchmark_results,
    save_detailed_report,
)
