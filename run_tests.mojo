"""Test runner for elementary cellular automaton tests.

Run with: pixi run mojo run_tests.mojo

Test documentation is in elementary/tests/*.mojo files.
"""

from testing import assert_true, assert_equal, assert_false
from elementary.grid import Grid
from elementary.rule import Rule
from shared.common import WIDTH, HEIGHT
from sys import has_accelerator
from sys.info import has_nvidia_gpu_accelerator


# ═══════════════════════════════════════════════════════════════════════════════
# TEST INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

struct TestStats:
    """Track test execution statistics."""
    var tests_run: Int
    var tests_passed: Int
    var tests_failed: Int
    
    fn __init__(out self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0


fn run_test(mut stats: TestStats, test_name: String, test_fn: fn() raises -> None):
    """Run a single test and track results."""
    stats.tests_run += 1
    try:
        test_fn()
        stats.tests_passed += 1
        print("  PASS:", test_name)
    except:
        stats.tests_failed += 1
        print("  FAIL:", test_name)


fn print_section(name: String):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)


fn print_summary(stats: TestStats):
    """Print test summary."""
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print("Tests run:", stats.tests_run)
    print("Tests passed:", stats.tests_passed)
    print("Tests failed:", stats.tests_failed)
    
    if stats.tests_failed == 0:
        print("\nALL TESTS PASSED!")
    else:
        print("\nSOME TESTS FAILED!")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

fn create_rule_110() -> Rule:
    """Create Rule 110 for testing."""
    var groups = List[List[String]]()
    var g1 = List[String]()
    g1.append("110")
    g1.append("101")
    groups.append(g1^)
    var g2 = List[String]()
    g2.append("011")
    g2.append("010")
    groups.append(g2^)
    var g3 = List[String]()
    g3.append("001")
    groups.append(g3^)
    return Rule("Rule 110", "rule_110", groups^)


fn create_rule_30() -> Rule:
    """Create Rule 30 for testing."""
    var groups = List[List[String]]()
    var g1 = List[String]()
    g1.append("100")
    g1.append("011")
    groups.append(g1^)
    var g2 = List[String]()
    g2.append("010")
    g2.append("001")
    groups.append(g2^)
    return Rule("Rule 30", "rule_30", groups^)


fn grids_equal(grid1: Grid, grid2: Grid) -> Bool:
    """Check if two grids are identical."""
    if grid1.width != grid2.width or grid1.height != grid2.height:
        return False
    for row in range(grid1.height):
        for col in range(grid1.width):
            if grid1.cells[row][col] != grid2.cells[row][col]:
                return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS: Grid (see elementary/tests/test_grid.mojo)
# ═══════════════════════════════════════════════════════════════════════════════

fn test_grid_dimensions() raises:
    var grid = Grid(WIDTH, HEIGHT)
    assert_equal(grid.get_width(), WIDTH)
    assert_equal(grid.get_height(), HEIGHT)


fn test_grid_initialized_with_zeros() raises:
    var grid = Grid(WIDTH, HEIGHT)
    for row in range(min(100, HEIGHT)):
        for col in range(min(100, WIDTH)):
            assert_equal(grid.cells[row][col], 0)


fn test_set_get_cell() raises:
    var grid = Grid(WIDTH, HEIGHT)
    grid.set_cell(0, 50, 1)
    assert_equal(grid.get_cell(0, 50), 1)


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS: CPU Methods (see elementary/tests/test_cpu_methods.mojo)
# ═══════════════════════════════════════════════════════════════════════════════

fn test_sequential_cpu_center_initialized() raises:
    var rule = create_rule_110()
    var grid = Grid(WIDTH, HEIGHT)
    grid.generate_sequential_cpu(rule)
    assert_equal(grid.cells[0][WIDTH // 2], 1)


fn test_sequential_cpu_edges_zero() raises:
    var rule = create_rule_110()
    var grid = Grid(WIDTH, HEIGHT)
    grid.generate_sequential_cpu(rule)
    for row in range(min(100, HEIGHT)):
        assert_equal(grid.cells[row][0], 0)
        assert_equal(grid.cells[row][WIDTH - 1], 0)


fn test_parallel_cpu_center_initialized() raises:
    var rule = create_rule_110()
    var grid = Grid(WIDTH, HEIGHT)
    grid.generate_parallel_cpu(rule)
    assert_equal(grid.cells[0][WIDTH // 2], 1)


fn test_sequential_parallel_identical() raises:
    var rule = create_rule_110()
    var grid_seq = Grid(WIDTH, HEIGHT)
    grid_seq.generate_sequential_cpu(rule)
    var grid_par = Grid(WIDTH, HEIGHT)
    grid_par.generate_parallel_cpu(rule)
    assert_true(grids_equal(grid_seq, grid_par))


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS: GPU Methods (see elementary/tests/test_gpu_methods.mojo)
# ═══════════════════════════════════════════════════════════════════════════════

fn test_native_gpu_executes() raises:
    @parameter
    if not has_accelerator():
        return
    var rule = create_rule_110()
    var grid = Grid(WIDTH, HEIGHT)
    var timing = grid.generate_native_gpu(rule)
    assert_true(timing.runs > 0)


fn test_cupy_gpu_executes() raises:
    if not has_nvidia_gpu_accelerator():
        return
    var rule = create_rule_110()
    var grid = Grid(WIDTH, HEIGHT)
    var timing = grid.generate_parallel_cells_cupy_gpu(rule)
    assert_true(timing.runs > 0)


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS: Rules (see elementary/tests/test_rules.mojo)
# ═══════════════════════════════════════════════════════════════════════════════

fn test_rule30_patterns() raises:
    var rule = create_rule_30()
    assert_equal(rule.apply(1, 0, 0), 1)
    assert_equal(rule.apply(0, 1, 1), 1)
    assert_equal(rule.apply(0, 1, 0), 1)
    assert_equal(rule.apply(0, 0, 1), 1)
    assert_equal(rule.apply(0, 0, 0), 0)
    assert_equal(rule.apply(1, 1, 1), 0)
    assert_equal(rule.apply(1, 1, 0), 0)
    assert_equal(rule.apply(1, 0, 1), 0)


fn test_rule110_patterns() raises:
    var rule = create_rule_110()
    assert_equal(rule.apply(1, 1, 0), 1)
    assert_equal(rule.apply(1, 0, 1), 1)
    assert_equal(rule.apply(0, 1, 1), 1)
    assert_equal(rule.apply(0, 1, 0), 1)
    assert_equal(rule.apply(0, 0, 1), 1)
    assert_equal(rule.apply(0, 0, 0), 0)
    assert_equal(rule.apply(1, 1, 1), 0)
    assert_equal(rule.apply(1, 0, 0), 0)


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS: Edge Cases (see elementary/tests/test_edge_cases.mojo)
# ═══════════════════════════════════════════════════════════════════════════════

fn test_empty_rule() raises:
    var empty_groups = List[List[String]]()
    var empty_rule = Rule("Empty", "empty", empty_groups^)
    var grid = Grid(WIDTH, HEIGHT)
    grid.generate_sequential_cpu(empty_rule)
    assert_equal(grid.cells[0][WIDTH // 2], 1)
    for col in range(WIDTH):
        assert_equal(grid.cells[1][col], 0)


fn test_all_patterns_rule() raises:
    var all_groups = List[List[String]]()
    var all_patterns = List[String]()
    all_patterns.append("000")
    all_patterns.append("001")
    all_patterns.append("010")
    all_patterns.append("011")
    all_patterns.append("100")
    all_patterns.append("101")
    all_patterns.append("110")
    all_patterns.append("111")
    all_groups.append(all_patterns^)
    var all_rule = Rule("All", "all", all_groups^)
    var grid = Grid(WIDTH, HEIGHT)
    grid.generate_sequential_cpu(all_rule)
    assert_equal(grid.cells[1][WIDTH // 2], 1)
    assert_equal(grid.cells[1][0], 0)
    assert_equal(grid.cells[1][WIDTH - 1], 0)


fn test_all_edges_zero() raises:
    var rule = create_rule_110()
    var grid = Grid(WIDTH, HEIGHT)
    grid.generate_sequential_cpu(rule)
    for row in range(HEIGHT):
        assert_equal(grid.cells[row][0], 0)
        assert_equal(grid.cells[row][WIDTH - 1], 0)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TEST RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

fn main() raises:
    print("=" * 60)
    print("ELEMENTARY CELLULAR AUTOMATON TEST SUITE")
    print("=" * 60)
    print("Grid dimensions:", WIDTH, "x", HEIGHT)
    
    var stats = TestStats()
    
    print_section("Grid Tests")
    run_test(stats, "test_grid_dimensions", test_grid_dimensions)
    run_test(stats, "test_grid_initialized_with_zeros", test_grid_initialized_with_zeros)
    run_test(stats, "test_set_get_cell", test_set_get_cell)
    
    print_section("CPU Method Tests")
    run_test(stats, "test_sequential_cpu_center_initialized", test_sequential_cpu_center_initialized)
    run_test(stats, "test_sequential_cpu_edges_zero", test_sequential_cpu_edges_zero)
    run_test(stats, "test_parallel_cpu_center_initialized", test_parallel_cpu_center_initialized)
    run_test(stats, "test_sequential_parallel_identical", test_sequential_parallel_identical)
    
    print_section("GPU Method Tests")
    run_test(stats, "test_native_gpu_executes", test_native_gpu_executes)
    run_test(stats, "test_cupy_gpu_executes", test_cupy_gpu_executes)
    
    print_section("Rule Tests")
    run_test(stats, "test_rule30_patterns", test_rule30_patterns)
    run_test(stats, "test_rule110_patterns", test_rule110_patterns)
    
    print_section("Edge Case Tests")
    run_test(stats, "test_empty_rule", test_empty_rule)
    run_test(stats, "test_all_patterns_rule", test_all_patterns_rule)
    run_test(stats, "test_all_edges_zero", test_all_edges_zero)
    
    print_summary(stats)

