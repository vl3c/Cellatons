"""Test runner for row cellular automaton tests.

Run with: pixi run mojo row/run_tests.mojo

Imports and runs tests from row/tests/ modules.
Also runs Python tests for the renderer module.
"""

from python import Python
from testing import TestSuite, assert_true, assert_equal, assert_false
from shared.common import WIDTH, HEIGHT

# Import all test functions from test modules
from row.tests.test_grid import (
    test_grid_dimensions,
    test_grid_initialized_with_zeros,
    test_set_get_cell,
    test_center_position_valid,
    test_stride_is_64_byte_aligned,
    test_stride_at_least_width,
    test_cell_access_across_rows,
    test_cell_access_at_row_end,
)

from row.tests.test_cpu_methods import (
    test_sequential_cpu_center_initialized,
    test_sequential_cpu_edges_zero,
    test_parallel_cpu_center_initialized,
    test_parallel_cpu_edges_zero,
    test_sequential_parallel_produce_identical_results,
    test_simd_cpu_center_initialized,
    test_simd_cpu_edges_zero,
    test_simd_matches_sequential,
)

from row.tests.test_gpu_methods import (
    test_native_gpu_executes_successfully,
    test_cupy_gpu_executes_successfully,
    test_pingpong_fallback_path_exists,
    test_native_gpu_benchmark_optimized,
    test_native_gpu_timing_reasonable,
)

from row.tests.test_rules import (
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

from row.tests.test_edge_cases import (
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
    print("Row Automata TEST SUITE")
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
    run_python_renderer_tests()


fn run_python_renderer_tests() raises:
    """Run Python renderer tests with formatted summary."""
    var sys = Python.import_module("sys")
    sys.path.insert(0, "row/renderer")
    sys.path.insert(0, "row/tests")
    
    var unittest = Python.import_module("unittest")
    var test_viewer = Python.import_module("test_viewer")
    var time = Python.import_module("time")
    var builtins = Python.import_module("builtins")
    
    var loader = unittest.TestLoader()
    var suite = loader.loadTestsFromModule(test_viewer)
    var TestSuite = unittest.TestSuite
    
    print()
    print("=" * 60)
    print()
    print("RENDERER TESTS (Python)")
    print()
    print("=" * 60)
    
    # Flatten suite into individual tests preserving order
    var stack = builtins.list()
    stack.append(suite)
    var tests_list = builtins.list()
    
    while Int(len(stack)) > 0:
        var item = stack.pop()
        if builtins.isinstance(item, TestSuite):
            var children = builtins.list(item)
            children.reverse()
            for child in children:
                stack.append(child)
        else:
            tests_list.append(item)
    
    var total_fail = 0
    var total_error = 0
    
    for test in tests_list:
        var result = unittest.TestResult()
        var start = time.time()
        test.run(result)
        var elapsed_py = time.time() - start  # Python float
        
        var full_name = test.id()
        var parts = full_name.rsplit(".", 1)
        var name = parts[1] if Int(len(parts)) > 1 else full_name
        
        var fmt = builtins.format
        var elapsed_str = fmt(elapsed_py, ".3f")
        
        if Int(len(result.failures)) > 0 or Int(len(result.errors)) > 0:
            total_fail += Int(len(result.failures))
            total_error += Int(len(result.errors))
            print("    FAIL [", elapsed_str, "]", name)
        else:
            print("    PASS [", elapsed_str, "]", name)
    
    if total_fail > 0 or total_error > 0:
        print("RENDERER TESTS FAILED:", total_fail, "failures,", total_error, "errors")
    else:
        print("--------")
        print("All renderer tests passed!\n\n")
