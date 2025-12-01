"""Test runner for elementary cellular automaton tests.

Run with: pixi run mojo run_tests.mojo

Uses TestSuite for automatic test discovery and execution.
"""

from testing import TestSuite, assert_true, assert_equal, assert_false
from elementary.grid import Grid
from elementary.rule import Rule
from shared.common import WIDTH, HEIGHT
from sys import has_accelerator
from sys.info import has_nvidia_gpu_accelerator


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
# TESTS: Grid
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
# TESTS: CPU Methods
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
# TESTS: GPU Methods
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
# TESTS: Rules
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
# TESTS: Edge Cases
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
# MAIN - Auto-discover and run all tests
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("ELEMENTARY CELLULAR AUTOMATON TEST SUITE")
    print("=" * 60)
    print("Grid dimensions:", WIDTH, "x", HEIGHT)
    print()
    
    TestSuite.discover_tests[__functions_in_module()]().run()
