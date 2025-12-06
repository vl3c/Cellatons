"""Tests for Grid Game of Life grid operations."""

from testing import assert_true, assert_equal
from grid.grid import Grid, DISPLAY_WIDTH, DISPLAY_HEIGHT, simd_width


# ─────────────────────────────────────────────────────────────────────────────
# Grid Initialization Tests
# ─────────────────────────────────────────────────────────────────────────────


fn test_grid_dimensions() raises:
    """Grid should have correct dimensions after initialization."""
    var grid = Grid(100, 50)
    assert_equal(grid.width, 100)
    assert_equal(grid.height, 50)


fn test_grid_initialized_with_zeros() raises:
    """Grid should be zero-initialized."""
    var grid = Grid(10, 10)
    for row in range(10):
        for col in range(10):
            assert_equal(grid.get_cell(row, col), 0)


fn test_stride_is_64_byte_aligned() raises:
    """Stride should be aligned to 64 bytes for AVX-512."""
    var grid = Grid(100, 50)
    assert_equal(grid.stride % simd_width, 0)


fn test_stride_at_least_width() raises:
    """Stride should be at least as large as width."""
    var grid = Grid(100, 50)
    assert_true(grid.stride >= grid.width)


fn test_small_grid_stride_alignment() raises:
    """Even small grids should have aligned stride."""
    var grid = Grid(10, 10)
    assert_equal(grid.stride % simd_width, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Cell Access Tests
# ─────────────────────────────────────────────────────────────────────────────


fn test_set_get_cell() raises:
    """Setting and getting a cell should work correctly."""
    var grid = Grid(10, 10)
    # Set in inactive buffer (set_cell writes to inactive)
    grid.set_cell(5, 5, 1)
    # Swap to make it active
    grid.swap_buffers()
    assert_equal(grid.get_cell(5, 5), 1)


fn test_cell_at_origin() raises:
    """Cell at (0,0) should be accessible."""
    var grid = Grid(10, 10)
    grid.set_cell(0, 0, 1)
    grid.swap_buffers()
    assert_equal(grid.get_cell(0, 0), 1)


fn test_cell_at_max_bounds() raises:
    """Cell at max bounds should be accessible."""
    var grid = Grid(10, 10)
    grid.set_cell(9, 9, 1)
    grid.swap_buffers()
    assert_equal(grid.get_cell(9, 9), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Toroidal Wrapping Tests
# ─────────────────────────────────────────────────────────────────────────────


fn test_toroidal_wrap_left() raises:
    """Accessing col=-1 should wrap to width-1."""
    var grid = Grid(10, 10)
    # Set cell at right edge
    grid.cells_a[grid._idx(5, 9)] = 1
    assert_equal(grid.get_cell_toroidal(5, -1), 1)


fn test_toroidal_wrap_right() raises:
    """Accessing col=width should wrap to 0."""
    var grid = Grid(10, 10)
    # Set cell at left edge
    grid.cells_a[grid._idx(5, 0)] = 1
    assert_equal(grid.get_cell_toroidal(5, 10), 1)


fn test_toroidal_wrap_top() raises:
    """Accessing row=-1 should wrap to height-1."""
    var grid = Grid(10, 10)
    # Set cell at bottom
    grid.cells_a[grid._idx(9, 5)] = 1
    assert_equal(grid.get_cell_toroidal(-1, 5), 1)


fn test_toroidal_wrap_bottom() raises:
    """Accessing row=height should wrap to 0."""
    var grid = Grid(10, 10)
    # Set cell at top
    grid.cells_a[grid._idx(0, 5)] = 1
    assert_equal(grid.get_cell_toroidal(10, 5), 1)


fn test_toroidal_wrap_corner() raises:
    """Accessing (-1,-1) should wrap to (height-1, width-1)."""
    var grid = Grid(10, 10)
    # Set cell at bottom-right corner
    grid.cells_a[grid._idx(9, 9)] = 1
    assert_equal(grid.get_cell_toroidal(-1, -1), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Neighbor Counting Tests
# ─────────────────────────────────────────────────────────────────────────────


fn test_count_neighbors_isolated_cell() raises:
    """Isolated cell should have 0 neighbors."""
    var grid = Grid(10, 10)
    grid.cells_a[grid._idx(5, 5)] = 1
    assert_equal(grid.count_neighbors(5, 5), 0)


fn test_count_neighbors_all_neighbors() raises:
    """Cell surrounded by 8 live cells should count 8 neighbors."""
    var grid = Grid(10, 10)
    # Set all 8 neighbors
    grid.cells_a[grid._idx(4, 4)] = 1
    grid.cells_a[grid._idx(4, 5)] = 1
    grid.cells_a[grid._idx(4, 6)] = 1
    grid.cells_a[grid._idx(5, 4)] = 1
    grid.cells_a[grid._idx(5, 6)] = 1
    grid.cells_a[grid._idx(6, 4)] = 1
    grid.cells_a[grid._idx(6, 5)] = 1
    grid.cells_a[grid._idx(6, 6)] = 1
    assert_equal(grid.count_neighbors(5, 5), 8)


fn test_count_neighbors_partial() raises:
    """Cell with 3 neighbors should count 3."""
    var grid = Grid(10, 10)
    grid.cells_a[grid._idx(4, 4)] = 1
    grid.cells_a[grid._idx(4, 5)] = 1
    grid.cells_a[grid._idx(5, 4)] = 1
    assert_equal(grid.count_neighbors(5, 5), 3)


fn test_count_neighbors_at_corner_with_wrapping() raises:
    """Corner cell should count neighbors with toroidal wrapping."""
    var grid = Grid(10, 10)
    # Set neighbors for (0,0) that wrap around
    grid.cells_a[grid._idx(9, 9)] = 1  # wraps to (-1,-1)
    grid.cells_a[grid._idx(9, 0)] = 1  # wraps to (-1,0)
    grid.cells_a[grid._idx(0, 9)] = 1  # wraps to (0,-1)
    assert_equal(grid.count_neighbors(0, 0), 3)


# ─────────────────────────────────────────────────────────────────────────────
# Buffer Swap Tests
# ─────────────────────────────────────────────────────────────────────────────


fn test_swap_buffers_toggles_active() raises:
    """Swapping buffers should toggle active flag."""
    var grid = Grid(10, 10)
    assert_equal(grid.active, 0)
    grid.swap_buffers()
    assert_equal(grid.active, 1)
    grid.swap_buffers()
    assert_equal(grid.active, 0)


fn test_double_buffering_isolation() raises:
    """Changes to inactive buffer shouldn't affect active reads."""
    var grid = Grid(10, 10)
    # Active is A (0), set in A
    grid.cells_a[grid._idx(5, 5)] = 1
    assert_equal(grid.get_cell(5, 5), 1)
    
    # Write to inactive (B)
    grid.set_cell(5, 5, 0)
    # Active still reads from A
    assert_equal(grid.get_cell(5, 5), 1)
    
    # After swap, B is active
    grid.swap_buffers()
    assert_equal(grid.get_cell(5, 5), 0)

