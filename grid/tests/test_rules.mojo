"""Tests for Grid Game of Life rules (B3/S23)."""

from testing import assert_true, assert_equal
from grid.grid import Grid


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────


fn _setup_grid_with_center_neighbors(neighbor_count: Int) -> Grid:
    """Create a grid with specified number of neighbors around center cell (5,5)."""
    var grid = Grid(10, 10)
    
    # Neighbor positions around (5,5) - set directly
    if neighbor_count >= 1:
        grid.cells_a[grid._idx(4, 4)] = 1
    if neighbor_count >= 2:
        grid.cells_a[grid._idx(4, 5)] = 1
    if neighbor_count >= 3:
        grid.cells_a[grid._idx(4, 6)] = 1
    if neighbor_count >= 4:
        grid.cells_a[grid._idx(5, 4)] = 1
    if neighbor_count >= 5:
        grid.cells_a[grid._idx(5, 6)] = 1
    if neighbor_count >= 6:
        grid.cells_a[grid._idx(6, 4)] = 1
    if neighbor_count >= 7:
        grid.cells_a[grid._idx(6, 5)] = 1
    if neighbor_count >= 8:
        grid.cells_a[grid._idx(6, 6)] = 1
    
    return grid^


fn _apply_grid_rules_single_cell(grid: Grid, row: Int, col: Int) -> Int:
    """Apply Grid rules to a single cell and return new state."""
    var neighbors = grid.count_neighbors(row, col)
    var current = grid.get_cell(row, col)
    
    if current == 1:
        # Live cell survives with 2 or 3 neighbors
        if neighbors == 2 or neighbors == 3:
            return 1
    else:
        # Dead cell becomes alive with exactly 3 neighbors
        if neighbors == 3:
            return 1
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Birth Rule Tests (B3 - Dead cell with exactly 3 neighbors becomes alive)
# ─────────────────────────────────────────────────────────────────────────────


fn test_birth_with_3_neighbors() raises:
    """Dead cell with exactly 3 neighbors should become alive."""
    var grid = _setup_grid_with_center_neighbors(3)
    # Center cell (5,5) is dead by default
    assert_equal(grid.get_cell(5, 5), 0)
    var result = _apply_grid_rules_single_cell(grid, 5, 5)
    assert_equal(result, 1)


fn test_no_birth_with_2_neighbors() raises:
    """Dead cell with 2 neighbors should stay dead."""
    var grid = _setup_grid_with_center_neighbors(2)
    var result = _apply_grid_rules_single_cell(grid, 5, 5)
    assert_equal(result, 0)


fn test_no_birth_with_4_neighbors() raises:
    """Dead cell with 4 neighbors should stay dead."""
    var grid = _setup_grid_with_center_neighbors(4)
    var result = _apply_grid_rules_single_cell(grid, 5, 5)
    assert_equal(result, 0)


fn test_no_birth_with_0_neighbors() raises:
    """Dead cell with 0 neighbors should stay dead."""
    var grid = Grid(10, 10)
    var result = _apply_grid_rules_single_cell(grid, 5, 5)
    assert_equal(result, 0)


fn test_no_birth_with_1_neighbor() raises:
    """Dead cell with 1 neighbor should stay dead."""
    var grid = _setup_grid_with_center_neighbors(1)
    var result = _apply_grid_rules_single_cell(grid, 5, 5)
    assert_equal(result, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Survival Rule Tests (S23 - Live cell survives with 2 or 3 neighbors)
# ─────────────────────────────────────────────────────────────────────────────


fn test_survival_with_2_neighbors() raises:
    """Live cell with 2 neighbors should survive."""
    var grid = _setup_grid_with_center_neighbors(2)
    grid.cells_a[grid._idx(5, 5)] = 1  # Make center cell alive
    var result = _apply_grid_rules_single_cell(grid, 5, 5)
    assert_equal(result, 1)


fn test_survival_with_3_neighbors() raises:
    """Live cell with 3 neighbors should survive."""
    var grid = _setup_grid_with_center_neighbors(3)
    grid.cells_a[grid._idx(5, 5)] = 1  # Make center cell alive
    var result = _apply_grid_rules_single_cell(grid, 5, 5)
    assert_equal(result, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Death Rule Tests (Underpopulation and Overpopulation)
# ─────────────────────────────────────────────────────────────────────────────


fn test_death_from_underpopulation_0_neighbors() raises:
    """Live cell with 0 neighbors should die (underpopulation)."""
    var grid = Grid(10, 10)
    grid.cells_a[grid._idx(5, 5)] = 1
    var result = _apply_grid_rules_single_cell(grid, 5, 5)
    assert_equal(result, 0)


fn test_death_from_underpopulation_1_neighbor() raises:
    """Live cell with 1 neighbor should die (underpopulation)."""
    var grid = _setup_grid_with_center_neighbors(1)
    grid.cells_a[grid._idx(5, 5)] = 1
    var result = _apply_grid_rules_single_cell(grid, 5, 5)
    assert_equal(result, 0)


fn test_death_from_overpopulation_4_neighbors() raises:
    """Live cell with 4 neighbors should die (overpopulation)."""
    var grid = _setup_grid_with_center_neighbors(4)
    grid.cells_a[grid._idx(5, 5)] = 1
    var result = _apply_grid_rules_single_cell(grid, 5, 5)
    assert_equal(result, 0)


fn test_death_from_overpopulation_5_neighbors() raises:
    """Live cell with 5 neighbors should die (overpopulation)."""
    var grid = _setup_grid_with_center_neighbors(5)
    grid.cells_a[grid._idx(5, 5)] = 1
    var result = _apply_grid_rules_single_cell(grid, 5, 5)
    assert_equal(result, 0)


fn test_death_from_overpopulation_8_neighbors() raises:
    """Live cell with 8 neighbors should die (overpopulation)."""
    var grid = _setup_grid_with_center_neighbors(8)
    grid.cells_a[grid._idx(5, 5)] = 1
    var result = _apply_grid_rules_single_cell(grid, 5, 5)
    assert_equal(result, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Classic Pattern Tests
# ─────────────────────────────────────────────────────────────────────────────


fn test_still_life_block() raises:
    """2x2 block should be stable (still life)."""
    var grid = Grid(10, 10)
    # Create 2x2 block at (4,4)
    grid.cells_a[grid._idx(4, 4)] = 1
    grid.cells_a[grid._idx(4, 5)] = 1
    grid.cells_a[grid._idx(5, 4)] = 1
    grid.cells_a[grid._idx(5, 5)] = 1
    
    # All 4 cells should survive (each has exactly 3 neighbors)
    assert_equal(_apply_grid_rules_single_cell(grid, 4, 4), 1)
    assert_equal(_apply_grid_rules_single_cell(grid, 4, 5), 1)
    assert_equal(_apply_grid_rules_single_cell(grid, 5, 4), 1)
    assert_equal(_apply_grid_rules_single_cell(grid, 5, 5), 1)


fn test_blinker_horizontal_to_vertical() raises:
    """Horizontal blinker should transition to vertical."""
    var grid = Grid(10, 10)
    # Create horizontal blinker at row 5
    grid.cells_a[grid._idx(5, 4)] = 1
    grid.cells_a[grid._idx(5, 5)] = 1
    grid.cells_a[grid._idx(5, 6)] = 1
    
    # Center cell should survive (2 neighbors)
    assert_equal(_apply_grid_rules_single_cell(grid, 5, 5), 1)
    
    # End cells should die (1 neighbor each)
    assert_equal(_apply_grid_rules_single_cell(grid, 5, 4), 0)
    assert_equal(_apply_grid_rules_single_cell(grid, 5, 6), 0)
    
    # Above and below center should be born (3 neighbors each)
    assert_equal(_apply_grid_rules_single_cell(grid, 4, 5), 1)
    assert_equal(_apply_grid_rules_single_cell(grid, 6, 5), 1)

