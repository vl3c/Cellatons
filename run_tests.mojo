"""Test runner for elementary cellular automaton tests.

Run with: pixi run mojo run_tests.mojo

Imports and runs tests from elementary/tests/ modules.
"""

from testing import TestSuite, assert_true, assert_equal, assert_false
from shared.common import WIDTH, HEIGHT

# Import all test functions from test modules
from elementary.tests.test_grid import (
    test_grid_dimensions,
    test_grid_initialized_with_zeros,
    test_set_get_cell,
    test_center_position_valid,
)

from elementary.tests.test_cpu_methods import (
    test_sequential_cpu_center_initialized,
    test_sequential_cpu_edges_zero,
    test_parallel_cpu_center_initialized,
    test_parallel_cpu_edges_zero,
    test_sequential_parallel_produce_identical_results,
)

from elementary.tests.test_gpu_methods import (
    test_native_gpu_executes_successfully,
    test_cupy_gpu_executes_successfully,
    test_pingpong_fallback_path_exists,
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
    
    # CPU method tests
    suite.test[test_sequential_cpu_center_initialized]()
    suite.test[test_sequential_cpu_edges_zero]()
    suite.test[test_parallel_cpu_center_initialized]()
    suite.test[test_parallel_cpu_edges_zero]()
    suite.test[test_sequential_parallel_produce_identical_results]()
    
    # GPU method tests
    suite.test[test_native_gpu_executes_successfully]()
    suite.test[test_cupy_gpu_executes_successfully]()
    suite.test[test_pingpong_fallback_path_exists]()
    
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
