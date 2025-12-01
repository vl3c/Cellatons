"""Unit tests for edge cases and boundary conditions.

Run with: mojo test elementary/tests/
"""

from testing import assert_true, assert_equal, assert_false
from elementary.grid import Grid
from elementary.rule import Rule
from shared.common import WIDTH, HEIGHT
from shared.logger import Logger


# ─────────────────────────────────────────────────────────────────────────────
# Edge Case: Empty Rule (no patterns produce 1)
# ─────────────────────────────────────────────────────────────────────────────

fn test_empty_rule_center_cell_set() raises:
    """Test that empty rule still sets center cell in first row."""
    var logger = Logger()
    var empty_groups = List[List[String]]()
    var empty_rule = Rule("Empty", "empty", empty_groups^)
    
    var grid = Grid(WIDTH, HEIGHT, logger)
    grid.generate_sequential_cpu(empty_rule)
    
    assert_equal(grid.get_cell(0, WIDTH // 2), 1)


fn test_empty_rule_row1_all_zeros() raises:
    """Test that empty rule produces all zeros in row 1."""
    var logger = Logger()
    var empty_groups = List[List[String]]()
    var empty_rule = Rule("Empty", "empty", empty_groups^)
    
    var grid = Grid(WIDTH, HEIGHT, logger)
    grid.generate_sequential_cpu(empty_rule)
    
    var row1_all_zero = True
    for col in range(WIDTH):
        if grid.get_cell(1, col) != 0:
            row1_all_zero = False
            break
    assert_true(row1_all_zero, "Row 1 should be all zeros with empty rule")


# ─────────────────────────────────────────────────────────────────────────────
# Edge Case: All-Ones Rule (all patterns produce 1)
# ─────────────────────────────────────────────────────────────────────────────

fn test_all_rule_row1_center_active() raises:
    """Test that all-ones rule activates row 1 center."""
    var logger = Logger()
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
    
    var grid = Grid(WIDTH, HEIGHT, logger)
    grid.generate_sequential_cpu(all_rule)
    
    assert_equal(grid.get_cell(1, WIDTH // 2), 1)


fn test_all_rule_edges_still_zero() raises:
    """Test that all-ones rule still keeps edges as zero."""
    var logger = Logger()
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
    
    var grid = Grid(WIDTH, HEIGHT, logger)
    grid.generate_sequential_cpu(all_rule)
    
    # Edges should still be 0 (boundary condition)
    assert_equal(grid.get_cell(1, 0), 0)
    assert_equal(grid.get_cell(1, WIDTH - 1), 0)


# ─────────────────────────────────────────────────────────────────────────────
# Boundary Tests
# ─────────────────────────────────────────────────────────────────────────────

fn create_rule_110() -> Rule:
    """Helper to create Rule 110."""
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


fn test_first_row_center_is_1() raises:
    """Test first row center cell is 1."""
    var logger = Logger()
    var rule = create_rule_110()
    var grid = Grid(WIDTH, HEIGHT, logger)
    grid.generate_sequential_cpu(rule)
    assert_equal(grid.get_cell(0, WIDTH // 2), 1)


fn test_first_row_left_edge_is_0() raises:
    """Test first row left edge is 0."""
    var logger = Logger()
    var rule = create_rule_110()
    var grid = Grid(WIDTH, HEIGHT, logger)
    grid.generate_sequential_cpu(rule)
    assert_equal(grid.get_cell(0, 0), 0)


fn test_first_row_right_edge_is_0() raises:
    """Test first row right edge is 0."""
    var logger = Logger()
    var rule = create_rule_110()
    var grid = Grid(WIDTH, HEIGHT, logger)
    grid.generate_sequential_cpu(rule)
    assert_equal(grid.get_cell(0, WIDTH - 1), 0)


fn test_last_row_left_edge_is_0() raises:
    """Test last row left edge is 0."""
    var logger = Logger()
    var rule = create_rule_110()
    var grid = Grid(WIDTH, HEIGHT, logger)
    grid.generate_sequential_cpu(rule)
    assert_equal(grid.get_cell(HEIGHT - 1, 0), 0)


fn test_last_row_right_edge_is_0() raises:
    """Test last row right edge is 0."""
    var logger = Logger()
    var rule = create_rule_110()
    var grid = Grid(WIDTH, HEIGHT, logger)
    grid.generate_sequential_cpu(rule)
    assert_equal(grid.get_cell(HEIGHT - 1, WIDTH - 1), 0)


fn test_all_rows_edges_are_0() raises:
    """Test that all rows have edge cells as 0."""
    var logger = Logger()
    var rule = create_rule_110()
    var grid = Grid(WIDTH, HEIGHT, logger)
    grid.generate_sequential_cpu(rule)
    
    var all_edges_zero = True
    for row in range(HEIGHT):
        if grid.get_cell(row, 0) != 0 or grid.get_cell(row, WIDTH - 1) != 0:
            all_edges_zero = False
            break
    assert_true(all_edges_zero, "All rows should have edge cells as 0")

