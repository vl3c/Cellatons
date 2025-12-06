"""Grid tests for the cube automaton."""

from testing import assert_equal, assert_true
from cube.grid import Grid, cube_WIDTH, cube_HEIGHT, cube_DEPTH


fn test_grid_dimensions() raises:
    var grid = Grid()
    assert_equal(grid.width, cube_WIDTH)
    assert_equal(grid.height, cube_HEIGHT)
    assert_equal(grid.depth, cube_DEPTH)


fn test_stride_alignment() raises:
    var grid = Grid()
    assert_true(grid.stride % 64 == 0)
    assert_true(grid.stride >= grid.width)


fn test_layer_stride_matches_height() raises:
    var grid = Grid()
    assert_equal(grid.layer_stride, grid.stride * grid.height)


fn test_neighbor_count_center() raises:
    var grid = Grid()
    var center_z = 2
    var center_y = 2
    var center_x = 2
    
    # Activate 6 neighbors around the center
    grid.cells_a[grid._idx(center_z, center_y, center_x - 1)] = 1
    grid.cells_a[grid._idx(center_z, center_y, center_x + 1)] = 1
    grid.cells_a[grid._idx(center_z, center_y - 1, center_x)] = 1
    grid.cells_a[grid._idx(center_z, center_y + 1, center_x)] = 1
    grid.cells_a[grid._idx(center_z - 1, center_y, center_x)] = 1
    grid.cells_a[grid._idx(center_z + 1, center_y, center_x)] = 1
    
    var count = grid.count_neighbors(center_z, center_y, center_x)
    assert_equal(count, 6)


fn test_neighbor_wraparound_edges() raises:
    var grid = Grid()
    # Set a single cell at the far corner; neighbor count at origin should see it due to wrap
    grid.cells_a[grid._idx(grid.depth - 1, grid.height - 1, grid.width - 1)] = 1
    
    var count = grid.count_neighbors(0, 0, 0)
    assert_equal(count, 1)


