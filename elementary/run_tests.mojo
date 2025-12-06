"""Test runner for elementary cellular automaton tests.

Run with: pixi run mojo elementary/run_tests.mojo

Imports and runs tests from elementary/tests/ modules.
Also runs Python tests for the renderer module.
"""

from python import Python
from testing import TestSuite, assert_true, assert_equal, assert_false
from shared.common import WIDTH, HEIGHT

# Import all test functions from test modules
from elementary.tests.test_grid import (
    test_grid_dimensions,
    test_grid_initialized_with_zeros,
    test_set_get_cell,
    test_center_position_valid,
    test_stride_is_64_byte_aligned,
    test_stride_at_least_width,
    test_cell_access_across_rows,
    test_cell_access_at_row_end,
)

from elementary.tests.test_cpu_methods import (
    test_sequential_cpu_center_initialized,
    test_sequential_cpu_edges_zero,
    test_parallel_cpu_center_initialized,
    test_parallel_cpu_edges_zero,
    test_sequential_parallel_produce_identical_results,
    test_simd_cpu_center_initialized,
    test_simd_cpu_edges_zero,
    test_simd_matches_sequential,
)

from elementary.tests.test_gpu_methods import (
    test_native_gpu_executes_successfully,
    test_cupy_gpu_executes_successfully,
    test_pingpong_fallback_path_exists,
    test_native_gpu_benchmark_optimized,
    test_native_gpu_timing_reasonable,
)

from elementary.tests.test_rules import (
    test_rule30_100_produces_1,
    test_rule30_011_produces_1,
    test_rule30_010_produces_1,
    test_rule30_001_produces_1,
    test_rule30_000_produces_0,
    test_rule30_111_produces_0,
    test_rule30_110_produces_0,
    test_rule30_101_produces_0,
    test_rule110_110_produces_1,
    test_rule110_101_produces_1,
    test_rule110_011_produces_1,
    test_rule110_010_produces_1,
    test_rule110_001_produces_1,
    test_rule110_000_produces_0,
    test_rule110_111_produces_0,
    test_rule110_100_produces_0,
)

from elementary.tests.test_edge_cases import (
    test_empty_rule_center_cell_set,
    test_empty_rule_row1_all_zeros,
    test_all_rule_row1_center_active,
    test_all_rule_edges_still_zero,
    test_first_row_center_is_1,
    test_first_row_left_edge_is_0,
    test_first_row_right_edge_is_0,
    test_last_row_left_edge_is_0,
    test_last_row_right_edge_is_0,
    test_all_rows_edges_are_0,
)


def main():
    print("=" * 60)
    print("ELEMENTARY CELLULAR AUTOMATON TEST SUITE")
    print("=" * 60)
    print("Grid dimensions:", WIDTH, "x", HEIGHT)
    print()
    
    var suite = TestSuite()
    
    # Grid tests
    suite.test[test_grid_dimensions]()
    suite.test[test_grid_initialized_with_zeros]()
    suite.test[test_set_get_cell]()
    suite.test[test_center_position_valid]()
    suite.test[test_stride_is_64_byte_aligned]()
    suite.test[test_stride_at_least_width]()
    suite.test[test_cell_access_across_rows]()
    suite.test[test_cell_access_at_row_end]()
    
    # CPU method tests
    suite.test[test_sequential_cpu_center_initialized]()
    suite.test[test_sequential_cpu_edges_zero]()
    suite.test[test_parallel_cpu_center_initialized]()
    suite.test[test_parallel_cpu_edges_zero]()
    suite.test[test_sequential_parallel_produce_identical_results]()
    suite.test[test_simd_cpu_center_initialized]()
    suite.test[test_simd_cpu_edges_zero]()
    suite.test[test_simd_matches_sequential]()
    
    # GPU method tests
    suite.test[test_native_gpu_executes_successfully]()
    suite.test[test_cupy_gpu_executes_successfully]()
    suite.test[test_pingpong_fallback_path_exists]()
    suite.test[test_native_gpu_benchmark_optimized]()
    suite.test[test_native_gpu_timing_reasonable]()
    
    # Rule 30 tests
    suite.test[test_rule30_100_produces_1]()
    suite.test[test_rule30_011_produces_1]()
    suite.test[test_rule30_010_produces_1]()
    suite.test[test_rule30_001_produces_1]()
    suite.test[test_rule30_000_produces_0]()
    suite.test[test_rule30_111_produces_0]()
    suite.test[test_rule30_110_produces_0]()
    suite.test[test_rule30_101_produces_0]()
    
    # Rule 110 tests
    suite.test[test_rule110_110_produces_1]()
    suite.test[test_rule110_101_produces_1]()
    suite.test[test_rule110_011_produces_1]()
    suite.test[test_rule110_010_produces_1]()
    suite.test[test_rule110_001_produces_1]()
    suite.test[test_rule110_000_produces_0]()
    suite.test[test_rule110_111_produces_0]()
    suite.test[test_rule110_100_produces_0]()
    
    # Edge case tests
    suite.test[test_empty_rule_center_cell_set]()
    suite.test[test_empty_rule_row1_all_zeros]()
    suite.test[test_all_rule_row1_center_active]()
    suite.test[test_all_rule_edges_still_zero]()
    suite.test[test_first_row_center_is_1]()
    suite.test[test_first_row_left_edge_is_0]()
    suite.test[test_first_row_right_edge_is_0]()
    suite.test[test_last_row_left_edge_is_0]()
    suite.test[test_last_row_right_edge_is_0]()
    suite.test[test_all_rows_edges_are_0]()
    
    suite^.run()
    
    # Run Python renderer tests
    print()
    print("=" * 60)
    print("RENDERER TESTS (Python)")
    print("=" * 60)
    run_python_renderer_tests()


fn run_python_renderer_tests() raises:
    """Run Python unittest tests for the renderer module."""
    var sys = Python.import_module("sys")
    sys.path.insert(0, "elementary/renderer")
    sys.path.insert(0, "elementary/tests")
    
    var unittest = Python.import_module("unittest")
    var test_viewer = Python.import_module("test_viewer")
    
    # Create test suite and run
    var loader = unittest.TestLoader()
    var suite = loader.loadTestsFromModule(test_viewer)
    var runner = unittest.TextTestRunner(verbosity=2)
    var result = runner.run(suite)
    
    # Check if all tests passed
    var failures = Int(len(result.failures))
    var errors = Int(len(result.errors))
    if failures > 0 or errors > 0:
        print("RENDERER TESTS FAILED:", failures, "failures,", errors, "errors")
    else:
        print("All renderer tests passed!")
