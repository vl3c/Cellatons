"""Unit tests for Grid initialization and basic operations.

Run with: mojo test elementary/tests/
"""

from testing import assert_true, assert_equal, assert_false
from elementary.grid import Grid
from elementary.rule import Rule
from shared.common import WIDTH, HEIGHT
from shared.logger import Logger


# ─────────────────────────────────────────────────────────────────────────────
# Grid Initialization Tests
# ─────────────────────────────────────────────────────────────────────────────

fn test_grid_dimensions() raises:
    """Test that grid dimensions match constants."""
    var logger = Logger()
    var grid = Grid(WIDTH, HEIGHT, logger)
    assert_equal(grid.get_width(), WIDTH)
    assert_equal(grid.get_height(), HEIGHT)


fn test_grid_initialized_with_zeros() raises:
    """Test that new grid is initialized with all zeros."""
    var logger = Logger()
    var grid = Grid(WIDTH, HEIGHT, logger)
    var all_zeros = True
    for row in range(min(100, HEIGHT)):
        for col in range(min(100, WIDTH)):
            if grid.get_cell(row, col) != 0:
                all_zeros = False
                break
    assert_true(all_zeros, "Grid should be initialized with all zeros")


fn test_set_get_cell() raises:
    """Test set_cell and get_cell methods."""
    var logger = Logger()
    var grid = Grid(WIDTH, HEIGHT, logger)
    grid.set_cell(0, 50, 1)
    assert_equal(grid.get_cell(0, 50), 1)
    
    grid.set_cell(5, 100, 1)
    assert_equal(grid.get_cell(5, 100), 1)
    
    # Verify other cells still 0
    assert_equal(grid.get_cell(0, 51), 0)


fn test_center_position_valid() raises:
    """Test that center position calculation is valid."""
    var center = WIDTH // 2
    assert_true(center > 0, "Center should be positive")
    assert_true(center < WIDTH, "Center should be less than WIDTH")

