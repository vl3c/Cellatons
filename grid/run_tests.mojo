"""Test runner for Grid Game of Life tests.

Run with: pixi run mojo grid/run_tests.mojo
"""

from python import Python, PythonObject
from testing import TestSuite

# Import test functions
from grid.tests.test_grid import (
    test_grid_dimensions,
    test_grid_initialized_with_zeros,
    test_stride_is_64_byte_aligned,
    test_stride_at_least_width,
    test_small_grid_stride_alignment,
    test_set_get_cell,
    test_cell_at_origin,
    test_cell_at_max_bounds,
    test_toroidal_wrap_left,
    test_toroidal_wrap_right,
    test_toroidal_wrap_top,
    test_toroidal_wrap_bottom,
    test_toroidal_wrap_corner,
    test_count_neighbors_isolated_cell,
    test_count_neighbors_all_neighbors,
    test_count_neighbors_partial,
    test_count_neighbors_at_corner_with_wrapping,
    test_swap_buffers_toggles_active,
    test_double_buffering_isolation,
)

from grid.tests.test_rules import (
    test_birth_with_3_neighbors,
    test_no_birth_with_2_neighbors,
    test_no_birth_with_4_neighbors,
    test_no_birth_with_0_neighbors,
    test_no_birth_with_1_neighbor,
    test_survival_with_2_neighbors,
    test_survival_with_3_neighbors,
    test_death_from_underpopulation_0_neighbors,
    test_death_from_underpopulation_1_neighbor,
    test_death_from_overpopulation_4_neighbors,
    test_death_from_overpopulation_5_neighbors,
    test_death_from_overpopulation_8_neighbors,
    test_still_life_block,
    test_blinker_horizontal_to_vertical,
)

from grid.tests.test_cpu_generation import (
    test_cpu_step_runs_without_error,
    test_cpu_step_swaps_buffers,
    test_empty_grid_stays_empty,
    test_isolated_cell_dies,
    test_block_stability,
    test_blinker_oscillation,
    test_blinker_period_2,
    test_toroidal_wrapping_horizontal,
    test_toroidal_wrapping_vertical,
    test_toroidal_corner_interaction,
    test_large_grid_generation,
    test_multiple_generations,
)

from grid.tests.test_gpu_generation import (
    test_has_gpu_returns_bool,
    test_gpu_detection_consistent,
    test_gpu_compute_initialization,
    test_gpu_compute_step_runs,
    test_gpu_compute_swaps_active,
    test_gpu_compute_upload_download_roundtrip,
    test_gpu_cpu_produce_same_result_empty_grid,
    test_gpu_cpu_produce_same_result_block,
    test_gpu_cpu_produce_same_result_blinker,
)

from grid.tests.test_benchmark import (
    test_stats_empty_count,
    test_stats_add_increments_count,
    test_stats_clear_resets_count,
    test_stats_avg_empty,
    test_stats_avg_single,
    test_stats_avg_multiple,
    test_stats_min_empty,
    test_stats_min_single,
    test_stats_min_multiple,
    test_stats_max_empty,
    test_stats_max_single,
    test_stats_max_multiple,
    test_stats_std_dev_empty,
    test_stats_std_dev_single,
    test_stats_std_dev_identical,
    test_stats_std_dev_varied,
    test_result_total_cells,
    test_result_preserves_dimensions,
)


def main():
    print("=" * 60)
    print("Grid Game of Life TEST SUITE")
    print("=" * 60)
    print()
    
    var suite = TestSuite()
    
    # Grid tests
    print("Grid Tests:")
    suite.test[test_grid_dimensions]()
    suite.test[test_grid_initialized_with_zeros]()
    suite.test[test_stride_is_64_byte_aligned]()
    suite.test[test_stride_at_least_width]()
    suite.test[test_small_grid_stride_alignment]()
    suite.test[test_set_get_cell]()
    suite.test[test_cell_at_origin]()
    suite.test[test_cell_at_max_bounds]()
    suite.test[test_toroidal_wrap_left]()
    suite.test[test_toroidal_wrap_right]()
    suite.test[test_toroidal_wrap_top]()
    suite.test[test_toroidal_wrap_bottom]()
    suite.test[test_toroidal_wrap_corner]()
    suite.test[test_count_neighbors_isolated_cell]()
    suite.test[test_count_neighbors_all_neighbors]()
    suite.test[test_count_neighbors_partial]()
    suite.test[test_count_neighbors_at_corner_with_wrapping]()
    suite.test[test_swap_buffers_toggles_active]()
    suite.test[test_double_buffering_isolation]()
    
    # Rule tests
    print("Rule Tests:")
    suite.test[test_birth_with_3_neighbors]()
    suite.test[test_no_birth_with_2_neighbors]()
    suite.test[test_no_birth_with_4_neighbors]()
    suite.test[test_no_birth_with_0_neighbors]()
    suite.test[test_no_birth_with_1_neighbor]()
    suite.test[test_survival_with_2_neighbors]()
    suite.test[test_survival_with_3_neighbors]()
    suite.test[test_death_from_underpopulation_0_neighbors]()
    suite.test[test_death_from_underpopulation_1_neighbor]()
    suite.test[test_death_from_overpopulation_4_neighbors]()
    suite.test[test_death_from_overpopulation_5_neighbors]()
    suite.test[test_death_from_overpopulation_8_neighbors]()
    suite.test[test_still_life_block]()
    suite.test[test_blinker_horizontal_to_vertical]()
    
    # CPU generation tests
    print("CPU Generation Tests:")
    suite.test[test_cpu_step_runs_without_error]()
    suite.test[test_cpu_step_swaps_buffers]()
    suite.test[test_empty_grid_stays_empty]()
    suite.test[test_isolated_cell_dies]()
    suite.test[test_block_stability]()
    suite.test[test_blinker_oscillation]()
    suite.test[test_blinker_period_2]()
    suite.test[test_toroidal_wrapping_horizontal]()
    suite.test[test_toroidal_wrapping_vertical]()
    suite.test[test_toroidal_corner_interaction]()
    suite.test[test_large_grid_generation]()
    suite.test[test_multiple_generations]()
    
    # GPU generation tests
    print("GPU Generation Tests:")
    suite.test[test_has_gpu_returns_bool]()
    suite.test[test_gpu_detection_consistent]()
    suite.test[test_gpu_compute_initialization]()
    suite.test[test_gpu_compute_step_runs]()
    suite.test[test_gpu_compute_swaps_active]()
    suite.test[test_gpu_compute_upload_download_roundtrip]()
    suite.test[test_gpu_cpu_produce_same_result_empty_grid]()
    suite.test[test_gpu_cpu_produce_same_result_block]()
    suite.test[test_gpu_cpu_produce_same_result_blinker]()
    
    # Benchmark tests
    print("Benchmark Tests:")
    suite.test[test_stats_empty_count]()
    suite.test[test_stats_add_increments_count]()
    suite.test[test_stats_clear_resets_count]()
    suite.test[test_stats_avg_empty]()
    suite.test[test_stats_avg_single]()
    suite.test[test_stats_avg_multiple]()
    suite.test[test_stats_min_empty]()
    suite.test[test_stats_min_single]()
    suite.test[test_stats_min_multiple]()
    suite.test[test_stats_max_empty]()
    suite.test[test_stats_max_single]()
    suite.test[test_stats_max_multiple]()
    suite.test[test_stats_std_dev_empty]()
    suite.test[test_stats_std_dev_single]()
    suite.test[test_stats_std_dev_identical]()
    suite.test[test_stats_std_dev_varied]()
    suite.test[test_result_total_cells]()
    suite.test[test_result_preserves_dimensions]()
    
    suite^.run()
    
    run_python_renderer_tests()


fn run_python_renderer_tests() raises:
    """Run Python renderer tests with formatted summary."""
    var sys = Python.import_module("sys")
    sys.path.insert(0, "grid/renderer")
    sys.path.insert(0, "grid/tests")
    
    var unittest = Python.import_module("unittest")
    var time = Python.import_module("time")
    var builtins = Python.import_module("builtins")
    
    print()
    print("=" * 60)
    print()
    print("RENDERER TESTS (Python)")
    print()
    print("=" * 60)
    
    # Attempt to import renderer tests; skip gracefully if absent
    var test_viewer: PythonObject
    try:
        test_viewer = Python.import_module("test_viewer")
    except e:
        print("    SKIP - no renderer tests found for grid")
        return
    
    var loader = unittest.TestLoader()
    var suite = loader.loadTestsFromModule(test_viewer)
    var TestSuite = unittest.TestSuite
    
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

