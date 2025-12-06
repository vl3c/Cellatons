"""Grid tests for the hypercube automaton."""

from testing import assert_equal, assert_true
from hypercube.grid import Grid, HYPER_WIDTH, HYPER_HEIGHT, HYPER_DEPTH, HYPER_W


fn test_grid_dimensions() raises:
    var grid = Grid()
    assert_equal(grid.width, HYPER_WIDTH)
    assert_equal(grid.height, HYPER_HEIGHT)
    assert_equal(grid.depth, HYPER_DEPTH)
    assert_equal(grid.w_dim, HYPER_W)


fn test_stride_alignment() raises:
    var grid = Grid()
    assert_true(grid.stride % 64 == 0)
    assert_true(grid.stride >= grid.width)


fn test_layer_and_hyperlayer_stride() raises:
    var grid = Grid()
    assert_equal(grid.layer_stride, grid.stride * grid.height)
    assert_equal(grid.hyperlayer_stride, grid.layer_stride * grid.depth)


fn test_neighbor_count_center() raises:
    var grid = Grid()
    var cw = 1
    var cz = 1
    var cy = 1
    var cx = 1
    
    # Activate 6 orthogonal neighbors around the center (X/Y/Z axes)
    grid.cells_a[grid._idx(cw, cz, cy, cx - 1)] = 1
    grid.cells_a[grid._idx(cw, cz, cy, cx + 1)] = 1
    grid.cells_a[grid._idx(cw, cz, cy - 1, cx)] = 1
    grid.cells_a[grid._idx(cw, cz, cy + 1, cx)] = 1
    grid.cells_a[grid._idx(cw, cz - 1, cy, cx)] = 1
    grid.cells_a[grid._idx(cw, cz + 1, cy, cx)] = 1
    
    var count = grid.count_neighbors(cw, cz, cy, cx)
    assert_equal(count, 6)


fn test_neighbor_wraparound_edges() raises:
    var grid = Grid()
    # Set a single cell at the far corner; neighbor count at origin should see it due to wrap
    grid.cells_a[grid._idx(grid.w_dim - 1, grid.depth - 1, grid.height - 1, grid.width - 1)] = 1
    
    var count = grid.count_neighbors(0, 0, 0, 0)
    assert_equal(count, 1)


fn test_neighbor_wraparound_w_axis() raises:
    """Wrap across W axis should be counted."""
    var grid = Grid()
    grid.cells_a[grid._idx(grid.w_dim - 1, 1, 1, 1)] = 1
    
    var count = grid.count_neighbors(0, 1, 1, 1)
    assert_equal(count, 1)


