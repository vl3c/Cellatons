"""Unit tests for Grid initialization and basic operations.

Run with: mojo test row/tests/
"""

from testing import assert_true, assert_equal, assert_false
from row.grid import Grid
from row.rule import Rule
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


# ─────────────────────────────────────────────────────────────────────────────
# Stride Alignment Tests
# ─────────────────────────────────────────────────────────────────────────────

fn test_stride_is_64_byte_aligned() raises:
    """Test that grid stride is aligned to 64 bytes for AVX-512."""
    var logger = Logger()
    var grid = Grid(WIDTH, HEIGHT, logger)
    assert_equal(grid.stride % 64, 0, "Stride should be 64-byte aligned")


fn test_stride_at_least_width() raises:
    """Test that stride is at least as large as width."""
    var logger = Logger()
    var grid = Grid(WIDTH, HEIGHT, logger)
    assert_true(grid.stride >= grid.width, "Stride should be >= width")


fn test_cell_access_across_rows() raises:
    """Test that cell access works correctly across row boundaries with aligned stride."""
    var logger = Logger()
    var grid = Grid(WIDTH, HEIGHT, logger)
    
    # Set cells in different rows
    grid.set_cell(0, 100, 1)
    grid.set_cell(1, 200, 1)
    grid.set_cell(2, 300, 1)
    
    # Verify they're independent and correct
    assert_equal(grid.get_cell(0, 100), 1)
    assert_equal(grid.get_cell(1, 200), 1)
    assert_equal(grid.get_cell(2, 300), 1)
    
    # Verify adjacent cells are still zero
    assert_equal(grid.get_cell(0, 101), 0)
    assert_equal(grid.get_cell(1, 201), 0)
    assert_equal(grid.get_cell(2, 301), 0)


fn test_cell_access_at_row_end() raises:
    """Test cell access near row end with stride padding."""
    var logger = Logger()
    var grid = Grid(WIDTH, HEIGHT, logger)
    
    # Set cell at end of row (within logical width)
    grid.set_cell(0, WIDTH - 2, 1)
    grid.set_cell(1, WIDTH - 2, 1)
    
    assert_equal(grid.get_cell(0, WIDTH - 2), 1)
    assert_equal(grid.get_cell(1, WIDTH - 2), 1)
    
    # Verify row 1 start is still zero (stride padding shouldn't leak)
    assert_equal(grid.get_cell(1, 0), 0)

