"""Tests for CPU SIMD generation in Conway's Game of Life."""

from testing import assert_true, assert_equal
from conway.grid import Grid
from conway.cpu_compute import CPUCompute


# ─────────────────────────────────────────────────────────────────────────────
# Basic Generation Tests
# ─────────────────────────────────────────────────────────────────────────────


fn test_cpu_step_runs_without_error() raises:
    """CPUCompute.step should execute without crashing."""
    var grid = Grid(100, 100)
    grid.cells_a[grid._idx(50, 50)] = 1
    CPUCompute.step(grid)
    # If we get here, no crash occurred
    assert_true(True)


fn test_cpu_step_swaps_buffers() raises:
    """CPUCompute.step should swap active buffer."""
    var grid = Grid(100, 100)
    assert_equal(grid.active, 0)
    CPUCompute.step(grid)
    assert_equal(grid.active, 1)


fn test_empty_grid_stays_empty() raises:
    """Empty grid should remain empty after generation."""
    var grid = Grid(50, 50)
    CPUCompute.step(grid)
    
    for row in range(50):
        for col in range(50):
            assert_equal(grid.get_cell(row, col), 0)


fn test_isolated_cell_dies() raises:
    """Single isolated cell should die from underpopulation."""
    var grid = Grid(50, 50)
    grid.cells_a[grid._idx(25, 25)] = 1
    CPUCompute.step(grid)
    assert_equal(grid.get_cell(25, 25), 0)


# ─────────────────────────────────────────────────────────────────────────────
# Pattern Evolution Tests
# ─────────────────────────────────────────────────────────────────────────────


fn test_block_stability() raises:
    """2x2 block should remain stable after generation."""
    var grid = Grid(50, 50)
    # Create 2x2 block
    grid.cells_a[grid._idx(24, 24)] = 1
    grid.cells_a[grid._idx(24, 25)] = 1
    grid.cells_a[grid._idx(25, 24)] = 1
    grid.cells_a[grid._idx(25, 25)] = 1
    
    CPUCompute.step(grid)
    
    # Block should still exist
    assert_equal(grid.get_cell(24, 24), 1)
    assert_equal(grid.get_cell(24, 25), 1)
    assert_equal(grid.get_cell(25, 24), 1)
    assert_equal(grid.get_cell(25, 25), 1)


fn test_blinker_oscillation() raises:
    """Horizontal blinker should become vertical after one step."""
    var grid = Grid(50, 50)
    # Create horizontal blinker
    grid.cells_a[grid._idx(25, 24)] = 1
    grid.cells_a[grid._idx(25, 25)] = 1
    grid.cells_a[grid._idx(25, 26)] = 1
    
    CPUCompute.step(grid)
    
    # Should now be vertical
    assert_equal(grid.get_cell(24, 25), 1)
    assert_equal(grid.get_cell(25, 25), 1)
    assert_equal(grid.get_cell(26, 25), 1)
    
    # Horizontal cells (except center) should be dead
    assert_equal(grid.get_cell(25, 24), 0)
    assert_equal(grid.get_cell(25, 26), 0)


fn test_blinker_period_2() raises:
    """Blinker should return to original state after 2 generations."""
    var grid = Grid(50, 50)
    # Create horizontal blinker
    grid.cells_a[grid._idx(25, 24)] = 1
    grid.cells_a[grid._idx(25, 25)] = 1
    grid.cells_a[grid._idx(25, 26)] = 1
    
    # Two steps should return to horizontal
    CPUCompute.step(grid)
    CPUCompute.step(grid)
    
    assert_equal(grid.get_cell(25, 24), 1)
    assert_equal(grid.get_cell(25, 25), 1)
    assert_equal(grid.get_cell(25, 26), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Toroidal Boundary Tests
# ─────────────────────────────────────────────────────────────────────────────


fn test_toroidal_wrapping_horizontal() raises:
    """Pattern at edge should interact across boundary."""
    var grid = Grid(50, 50)
    # Place cells at right edge that should birth cell at left edge
    # Need 3 neighbors at (25, 0) from cells at (24, 49), (25, 49), (26, 49)
    grid.cells_a[grid._idx(24, 49)] = 1
    grid.cells_a[grid._idx(25, 49)] = 1
    grid.cells_a[grid._idx(26, 49)] = 1
    
    CPUCompute.step(grid)
    
    # Cell at (25, 0) should be born (3 neighbors from wrapping)
    assert_equal(grid.get_cell(25, 0), 1)


fn test_toroidal_wrapping_vertical() raises:
    """Pattern at top/bottom should interact across boundary."""
    var grid = Grid(50, 50)
    # Place cells at bottom edge that should birth cell at top edge
    grid.cells_a[grid._idx(49, 24)] = 1
    grid.cells_a[grid._idx(49, 25)] = 1
    grid.cells_a[grid._idx(49, 26)] = 1
    
    CPUCompute.step(grid)
    
    # Cell at (0, 25) should be born
    assert_equal(grid.get_cell(0, 25), 1)


fn test_toroidal_corner_interaction() raises:
    """Cells at opposite corners should interact."""
    var grid = Grid(10, 10)
    # Place pattern that wraps around corner
    # Cell at (0,0) with neighbors at (9,9), (9,0), (0,9)
    grid.cells_a[grid._idx(9, 9)] = 1
    grid.cells_a[grid._idx(9, 0)] = 1
    grid.cells_a[grid._idx(0, 9)] = 1
    
    CPUCompute.step(grid)
    
    # (0,0) should be born with 3 neighbors
    assert_equal(grid.get_cell(0, 0), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Performance Sanity Tests
# ─────────────────────────────────────────────────────────────────────────────


fn test_large_grid_generation() raises:
    """Large grid should complete generation without issues."""
    var grid = Grid(500, 500)
    # Add some cells
    for i in range(100):
        grid.cells_a[grid._idx(250 + i % 10, 250 + i // 10)] = 1
    
    CPUCompute.step(grid)
    # If we get here, large grid generation works
    assert_true(True)


fn test_multiple_generations() raises:
    """Grid should handle multiple consecutive generations."""
    var grid = Grid(100, 100)
    # Create a glider
    grid.cells_a[grid._idx(50, 51)] = 1
    grid.cells_a[grid._idx(51, 52)] = 1
    grid.cells_a[grid._idx(52, 50)] = 1
    grid.cells_a[grid._idx(52, 51)] = 1
    grid.cells_a[grid._idx(52, 52)] = 1
    
    # Run 100 generations
    for _ in range(100):
        CPUCompute.step(grid)
    
    # Grid should still have exactly 5 live cells (glider is stable)
    var count = 0
    for row in range(100):
        for col in range(100):
            count += grid.get_cell(row, col)
    assert_equal(count, 5)
