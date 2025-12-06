"""GPU compute tests for the voxel automaton."""

from testing import assert_true, assert_equal
from sys import has_accelerator
from voxel.grid import Grid


fn test_has_gpu_returns_bool() raises:
    var grid = Grid()
    var result = grid.has_gpu()
    assert_true(result == True or result == False)


fn test_gpu_compute_initialization() raises:
    @parameter
    if has_accelerator():
        from voxel.gpu_compute import GPUCompute
        var grid = Grid()
        var compute = GPUCompute(grid.width, grid.height, grid.depth, grid.stride, grid.layer_stride)
        assert_equal(compute.width, grid.width)
        assert_equal(compute.height, grid.height)
        assert_equal(compute.depth, grid.depth)
        assert_equal(compute.stride, grid.stride)


fn test_gpu_compute_swaps_active() raises:
    @parameter
    if has_accelerator():
        from voxel.gpu_compute import GPUCompute
        var grid = Grid()
        var compute = GPUCompute(grid.width, grid.height, grid.depth, grid.stride, grid.layer_stride)
        assert_equal(compute.gpu_active, 0)
        compute.step()
        assert_equal(compute.gpu_active, 1)
        compute.step()
        assert_equal(compute.gpu_active, 0)


fn test_gpu_rule_birth_b6() raises:
    @parameter
    if has_accelerator():
        from voxel.gpu_compute import GPUCompute
        
        var grid = Grid()
        var center_z = 2
        var center_y = 2
        var center_x = 2
        
        # Six orthogonal neighbors alive, center dead
        grid.cells_a[grid._idx(center_z, center_y, center_x - 1)] = 1
        grid.cells_a[grid._idx(center_z, center_y, center_x + 1)] = 1
        grid.cells_a[grid._idx(center_z, center_y - 1, center_x)] = 1
        grid.cells_a[grid._idx(center_z, center_y + 1, center_x)] = 1
        grid.cells_a[grid._idx(center_z - 1, center_y, center_x)] = 1
        grid.cells_a[grid._idx(center_z + 1, center_y, center_x)] = 1
        
        var compute = GPUCompute(grid.width, grid.height, grid.depth, grid.stride, grid.layer_stride)
        var src_ptr = grid.cells_a.unsafe_ptr()
        compute.upload_from_cpu(src_ptr, 0)
        compute.step()
        compute.download_to_cpu(grid.cells_b)
        grid.active = compute.gpu_active
        
        assert_equal(grid.get_cell(center_z, center_y, center_x), 1)


fn test_gpu_wrap_across_faces_birth() raises:
    """Birth at origin using neighbors that wrap across opposite faces."""
    @parameter
    if has_accelerator():
        from voxel.gpu_compute import GPUCompute
        
        var grid = Grid()
        var max_z = grid.depth - 1
        var max_y = grid.height - 1
        var max_x = grid.width - 1
        
        # Place six neighbors around (0,0,0) using wrap on each axis
        grid.cells_a[grid._idx(0, 0, 1)] = 1       # +x
        grid.cells_a[grid._idx(0, 1, 0)] = 1       # +y
        grid.cells_a[grid._idx(1, 0, 0)] = 1       # +z
        grid.cells_a[grid._idx(0, 0, max_x)] = 1   # -x via wrap
        grid.cells_a[grid._idx(0, max_y, 0)] = 1   # -y via wrap
        grid.cells_a[grid._idx(max_z, 0, 0)] = 1   # -z via wrap
        
        var compute = GPUCompute(grid.width, grid.height, grid.depth, grid.stride, grid.layer_stride)
        var src_ptr = grid.cells_a.unsafe_ptr()
        compute.upload_from_cpu(src_ptr, 0)
        compute.step()
        compute.download_to_cpu(grid.cells_b)
        grid.active = compute.gpu_active
        
        # Birth at origin should occur (B6 rule)
        assert_equal(grid.get_cell(0, 0, 0), 1)


