"""Unit tests for CPU generation methods.

Run with: mojo test elementary/tests/
"""

from testing import assert_true, assert_equal, assert_false
from elementary.grid import Grid
from elementary.rule import Rule
from shared.common import WIDTH, HEIGHT
from shared.logger import Logger


fn create_rule_110() -> Rule:
    """Helper to create Rule 110 for testing."""
    var rule110_groups = List[List[String]]()
    var g1 = List[String]()
    g1.append("110")
    g1.append("101")
    rule110_groups.append(g1^)
    var g2 = List[String]()
    g2.append("011")
    g2.append("010")
    rule110_groups.append(g2^)
    var g3 = List[String]()
    g3.append("001")
    rule110_groups.append(g3^)
    return Rule("Rule 110", "rule_110", rule110_groups^)


fn grids_equal(grid1: Grid, grid2: Grid) -> Bool:
    """Check if two grids have identical cell values."""
    if grid1.width != grid2.width or grid1.height != grid2.height:
        return False
    for row in range(grid1.height):
        for col in range(grid1.width):
            if grid1.cells[row][col] != grid2.cells[row][col]:
                return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# CPU Sequential Tests
# ─────────────────────────────────────────────────────────────────────────────

fn test_sequential_cpu_center_initialized() raises:
    """Test that sequential CPU initializes center cell."""
    var logger = Logger()
    var rule = create_rule_110()
    var grid = Grid(WIDTH, HEIGHT, logger)
    grid.generate_sequential_cpu(rule)
    assert_equal(grid.cells[0][WIDTH // 2], 1)


fn test_sequential_cpu_edges_zero() raises:
    """Test that sequential CPU keeps edge cells as zero."""
    var logger = Logger()
    var rule = create_rule_110()
    var grid = Grid(WIDTH, HEIGHT, logger)
    grid.generate_sequential_cpu(rule)
    
    for row in range(min(100, HEIGHT)):
        assert_equal(grid.cells[row][0], 0)
        assert_equal(grid.cells[row][WIDTH - 1], 0)


# ─────────────────────────────────────────────────────────────────────────────
# CPU Parallel Tests
# ─────────────────────────────────────────────────────────────────────────────

fn test_parallel_cpu_center_initialized() raises:
    """Test that parallel CPU initializes center cell."""
    var logger = Logger()
    var rule = create_rule_110()
    var grid = Grid(WIDTH, HEIGHT, logger)
    grid.generate_parallel_cpu(rule)
    assert_equal(grid.cells[0][WIDTH // 2], 1)


fn test_parallel_cpu_edges_zero() raises:
    """Test that parallel CPU keeps edge cells as zero."""
    var logger = Logger()
    var rule = create_rule_110()
    var grid = Grid(WIDTH, HEIGHT, logger)
    grid.generate_parallel_cpu(rule)
    
    for row in range(min(100, HEIGHT)):
        assert_equal(grid.cells[row][0], 0)
        assert_equal(grid.cells[row][WIDTH - 1], 0)


# ─────────────────────────────────────────────────────────────────────────────
# CPU Consistency Tests
# ─────────────────────────────────────────────────────────────────────────────

fn test_sequential_parallel_produce_identical_results() raises:
    """Test that sequential and parallel CPU produce identical results."""
    var logger = Logger()
    var rule = create_rule_110()
    
    var grid_seq = Grid(WIDTH, HEIGHT, logger)
    grid_seq.generate_sequential_cpu(rule)
    
    var grid_par = Grid(WIDTH, HEIGHT, logger)
    grid_par.generate_parallel_cpu(rule)
    
    assert_true(grids_equal(grid_seq, grid_par), "Sequential and parallel should produce identical results")

