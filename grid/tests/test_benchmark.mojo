"""Tests for benchmark utilities."""

from testing import assert_true, assert_equal
from grid.benchmark import BenchmarkStats, BenchmarkResult


# ─────────────────────────────────────────────────────────────────────────────
# BenchmarkStats Tests
# ─────────────────────────────────────────────────────────────────────────────


fn test_stats_empty_count() raises:
    """Empty stats should have count 0."""
    var stats = BenchmarkStats("test")
    assert_equal(stats.count(), 0)


fn test_stats_add_increments_count() raises:
    """Adding samples should increment count."""
    var stats = BenchmarkStats("test")
    stats.add(1.0)
    assert_equal(stats.count(), 1)
    stats.add(2.0)
    assert_equal(stats.count(), 2)


fn test_stats_clear_resets_count() raises:
    """Clear should reset count to 0."""
    var stats = BenchmarkStats("test")
    stats.add(1.0)
    stats.add(2.0)
    stats.clear()
    assert_equal(stats.count(), 0)


fn test_stats_avg_empty() raises:
    """Average of empty stats should be 0."""
    var stats = BenchmarkStats("test")
    assert_equal(stats.avg(), 0.0)


fn test_stats_avg_single() raises:
    """Average of single value should be that value."""
    var stats = BenchmarkStats("test")
    stats.add(5.0)
    assert_equal(stats.avg(), 5.0)


fn test_stats_avg_multiple() raises:
    """Average should be correct for multiple values."""
    var stats = BenchmarkStats("test")
    stats.add(1.0)
    stats.add(2.0)
    stats.add(3.0)
    assert_equal(stats.avg(), 2.0)


fn test_stats_min_empty() raises:
    """Min of empty stats should be 0."""
    var stats = BenchmarkStats("test")
    assert_equal(stats.min_val(), 0.0)


fn test_stats_min_single() raises:
    """Min of single value should be that value."""
    var stats = BenchmarkStats("test")
    stats.add(5.0)
    assert_equal(stats.min_val(), 5.0)


fn test_stats_min_multiple() raises:
    """Min should find smallest value."""
    var stats = BenchmarkStats("test")
    stats.add(3.0)
    stats.add(1.0)
    stats.add(2.0)
    assert_equal(stats.min_val(), 1.0)


fn test_stats_max_empty() raises:
    """Max of empty stats should be 0."""
    var stats = BenchmarkStats("test")
    assert_equal(stats.max_val(), 0.0)


fn test_stats_max_single() raises:
    """Max of single value should be that value."""
    var stats = BenchmarkStats("test")
    stats.add(5.0)
    assert_equal(stats.max_val(), 5.0)


fn test_stats_max_multiple() raises:
    """Max should find largest value."""
    var stats = BenchmarkStats("test")
    stats.add(1.0)
    stats.add(3.0)
    stats.add(2.0)
    assert_equal(stats.max_val(), 3.0)


fn test_stats_std_dev_empty() raises:
    """Std dev of empty stats should be 0."""
    var stats = BenchmarkStats("test")
    assert_equal(stats.std_dev(), 0.0)


fn test_stats_std_dev_single() raises:
    """Std dev of single value should be 0."""
    var stats = BenchmarkStats("test")
    stats.add(5.0)
    assert_equal(stats.std_dev(), 0.0)


fn test_stats_std_dev_identical() raises:
    """Std dev of identical values should be 0."""
    var stats = BenchmarkStats("test")
    stats.add(5.0)
    stats.add(5.0)
    stats.add(5.0)
    assert_equal(stats.std_dev(), 0.0)


fn test_stats_std_dev_varied() raises:
    """Std dev should be non-zero for varied values."""
    var stats = BenchmarkStats("test")
    stats.add(1.0)
    stats.add(2.0)
    stats.add(3.0)
    assert_true(stats.std_dev() > 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# BenchmarkResult Tests
# ─────────────────────────────────────────────────────────────────────────────


fn test_result_total_cells() raises:
    """BenchmarkResult should calculate total cells correctly."""
    var gpu_stats = BenchmarkStats("GPU")
    var cpu_stats = BenchmarkStats("CPU")
    var frame_stats = BenchmarkStats("Frame")
    
    var result = BenchmarkResult(
        gpu_stats^, cpu_stats^, frame_stats^,
        100, 50
    )
    
    assert_equal(result.total_cells, 5000)


fn test_result_preserves_dimensions() raises:
    """BenchmarkResult should preserve grid dimensions."""
    var gpu_stats = BenchmarkStats("GPU")
    var cpu_stats = BenchmarkStats("CPU")
    var frame_stats = BenchmarkStats("Frame")
    
    var result = BenchmarkResult(
        gpu_stats^, cpu_stats^, frame_stats^,
        2560, 1440
    )
    
    assert_equal(result.grid_width, 2560)
    assert_equal(result.grid_height, 1440)

