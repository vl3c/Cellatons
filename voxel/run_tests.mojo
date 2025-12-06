"""Test runner for voxel automaton tests.

Run with: pixi run mojo voxel/run_tests.mojo
"""

from python import Python
from testing import TestSuite

from voxel.tests.test_grid import (
    test_grid_dimensions,
    test_stride_alignment,
    test_layer_stride_matches_height,
    test_neighbor_count_center,
    test_neighbor_wraparound_edges,
)

from voxel.tests.test_rules import (
    test_birth_on_six_neighbors,
    test_no_birth_on_five_or_seven,
    test_survival_on_five_six_seven,
    test_death_otherwise,
    test_neighbor_count_full_shell,
)

from voxel.tests.test_gpu import (
    test_has_gpu_returns_bool,
    test_gpu_compute_initialization,
    test_gpu_compute_swaps_active,
    test_gpu_rule_birth_b6,
    test_gpu_wrap_across_faces_birth,
)


def main():
    print("=" * 60)
    print("VOXEL AUTOMATON TEST SUITE")
    print("=" * 60)
    print()
    
    var suite = TestSuite()
    
    # Grid tests
    print("Grid Tests:")
    suite.test[test_grid_dimensions]()
    suite.test[test_stride_alignment]()
    suite.test[test_layer_stride_matches_height]()
    suite.test[test_neighbor_count_center]()
    suite.test[test_neighbor_wraparound_edges]()
    
    # Rule tests
    print("Rule Tests:")
    suite.test[test_birth_on_six_neighbors]()
    suite.test[test_no_birth_on_five_or_seven]()
    suite.test[test_survival_on_five_six_seven]()
    suite.test[test_death_otherwise]()
    suite.test[test_neighbor_count_full_shell]()
    
    # GPU tests (guarded in test bodies)
    print("GPU Tests:")
    suite.test[test_has_gpu_returns_bool]()
    suite.test[test_gpu_compute_initialization]()
    suite.test[test_gpu_compute_swaps_active]()
    suite.test[test_gpu_rule_birth_b6]()
    suite.test[test_gpu_wrap_across_faces_birth]()
    
    suite^.run()
    
    # Python renderer tests
    print()
    print("=" * 60)
    print("RENDERER TESTS (Python)")
    print("=" * 60)
    run_python_renderer_tests()


fn run_python_renderer_tests() raises:
    var sys = Python.import_module("sys")
    sys.path.insert(0, "voxel/renderer")
    sys.path.insert(0, "voxel/tests")
    
    var unittest = Python.import_module("unittest")
    var test_viewer = Python.import_module("test_viewer")
    var time = Python.import_module("time")
    var builtins = Python.import_module("builtins")
    
    var loader = unittest.TestLoader()
    var suite = loader.loadTestsFromModule(test_viewer)
    var TestSuite = unittest.TestSuite
    
    var total_fail = 0
    var total_error = 0
    
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
        print("All renderer tests passed!")


