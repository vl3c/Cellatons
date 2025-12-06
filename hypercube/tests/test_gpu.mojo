"""GPU compute tests for the hypercube automaton."""

from testing import assert_true, assert_equal
from sys import has_accelerator
from hypercube.grid import Grid


fn test_has_gpu_returns_bool() raises:
    var grid = Grid()
    var result = grid.has_gpu()
    assert_true(result == True or result == False)


fn test_gpu_compute_initialization() raises:
    @parameter
    if has_accelerator():
        from hypercube.gpu_compute import GPUCompute
        var grid = Grid()
        var compute = GPUCompute(
            grid.width,
            grid.height,
            grid.depth,
            grid.w_dim,
            grid.stride,
            grid.layer_stride,
            grid.hyperlayer_stride,
        )
        assert_equal(compute.width, grid.width)
        assert_equal(compute.height, grid.height)
        assert_equal(compute.depth, grid.depth)
        assert_equal(compute.w_dim, grid.w_dim)
        assert_equal(compute.stride, grid.stride)
        assert_equal(compute.layer_stride, grid.layer_stride)
        assert_equal(compute.hyperlayer_stride, grid.hyperlayer_stride)


fn test_gpu_compute_swaps_active() raises:
    @parameter
    if has_accelerator():
        from hypercube.gpu_compute import GPUCompute
        var grid = Grid()
        var compute = GPUCompute(
            grid.width,
            grid.height,
            grid.depth,
            grid.w_dim,
            grid.stride,
            grid.layer_stride,
            grid.hyperlayer_stride,
        )
        assert_equal(compute.gpu_active, 0)
        compute.step()
        assert_equal(compute.gpu_active, 1)
        compute.step()
        assert_equal(compute.gpu_active, 0)


fn test_gpu_rule_birth_b6() raises:
    @parameter
    if has_accelerator():
        from hypercube.gpu_compute import GPUCompute
        
        var grid = Grid()
        var cw = 1
        var cz = 1
        var cy = 1
        var cx = 1
        
        # Six orthogonal neighbors alive (X/Y/Z axes); center dead
        grid.cells_a[grid._idx(cw, cz, cy, cx - 1)] = 1
        grid.cells_a[grid._idx(cw, cz, cy, cx + 1)] = 1
        grid.cells_a[grid._idx(cw, cz, cy - 1, cx)] = 1
        grid.cells_a[grid._idx(cw, cz, cy + 1, cx)] = 1
        grid.cells_a[grid._idx(cw, cz - 1, cy, cx)] = 1
        grid.cells_a[grid._idx(cw, cz + 1, cy, cx)] = 1
        
        var compute = GPUCompute(
            grid.width,
            grid.height,
            grid.depth,
            grid.w_dim,
            grid.stride,
            grid.layer_stride,
            grid.hyperlayer_stride,
        )
        var src_ptr = grid.cells_a.unsafe_ptr()
        compute.upload_from_cpu(src_ptr, 0)
        compute.step()
        compute.download_to_cpu(grid.cells_b)
        grid.active = compute.gpu_active
        
        assert_equal(grid.get_cell(cw, cz, cy, cx), 1)


fn test_gpu_rule_birth_b6_w_axis() raises:
    """Birth using neighbors along W axis."""
    @parameter
    if has_accelerator():
        from hypercube.gpu_compute import GPUCompute
        
        var grid = Grid()
        var cw = 0
        var cz = 1
        var cy = 1
        var cx = 1
        
        # Six neighbors including +/-W and +/-X/Y
        grid.cells_a[grid._idx(cw, cz, cy, cx - 1)] = 1
        grid.cells_a[grid._idx(cw, cz, cy, cx + 1)] = 1
        grid.cells_a[grid._idx(cw, cz, cy - 1, cx)] = 1
        grid.cells_a[grid._idx(cw, cz, cy + 1, cx)] = 1
        grid.cells_a[grid._idx(1, cz, cy, cx)] = 1       # +w
        grid.cells_a[grid._idx(grid.w_dim - 1, cz, cy, cx)] = 1  # -w via wrap
        
        var compute = GPUCompute(
            grid.width,
            grid.height,
            grid.depth,
            grid.w_dim,
            grid.stride,
            grid.layer_stride,
            grid.hyperlayer_stride,
        )
        var src_ptr = grid.cells_a.unsafe_ptr()
        compute.upload_from_cpu(src_ptr, 0)
        compute.step()
        compute.download_to_cpu(grid.cells_b)
        grid.active = compute.gpu_active
        
        assert_equal(grid.get_cell(cw, cz, cy, cx), 1)


fn test_gpu_rule_no_birth_on_five_or_seven() raises:
    """Ensure births only at exactly six neighbors."""
    @parameter
    if has_accelerator():
        from hypercube.gpu_compute import GPUCompute
        
        # Case: five neighbors -> no birth
        var grid = Grid()
        var cw = 0
        var cz = 1
        var cy = 1
        var cx = 1
        grid.cells_a[grid._idx(cw, cz, cy, cx - 1)] = 1
        grid.cells_a[grid._idx(cw, cz, cy, cx + 1)] = 1
        grid.cells_a[grid._idx(cw, cz, cy - 1, cx)] = 1
        grid.cells_a[grid._idx(cw, cz, cy + 1, cx)] = 1
        grid.cells_a[grid._idx(1, cz, cy, cx)] = 1  # +w (5th)
        
        var compute = GPUCompute(
            grid.width,
            grid.height,
            grid.depth,
            grid.w_dim,
            grid.stride,
            grid.layer_stride,
            grid.hyperlayer_stride,
        )
        compute.upload_from_cpu(grid.cells_a.unsafe_ptr(), 0)
        compute.step()
        compute.download_to_cpu(grid.cells_b)
        grid.active = compute.gpu_active
        assert_equal(grid.get_cell(cw, cz, cy, cx), 0)
        
        # Case: seven neighbors -> no birth
        grid.randomize(0.0)
        grid.cells_a[grid._idx(cw, cz, cy, cx - 1)] = 1
        grid.cells_a[grid._idx(cw, cz, cy, cx + 1)] = 1
        grid.cells_a[grid._idx(cw, cz, cy - 1, cx)] = 1
        grid.cells_a[grid._idx(cw, cz, cy + 1, cx)] = 1
        grid.cells_a[grid._idx(1, cz, cy, cx)] = 1
        grid.cells_a[grid._idx(grid.w_dim - 1, cz, cy, cx)] = 1
        grid.cells_a[grid._idx(cw, cz + 1, cy, cx)] = 1  # add 7th
        
        compute.upload_from_cpu(grid.cells_a.unsafe_ptr(), 0)
        compute.step()
        compute.download_to_cpu(grid.cells_b)
        grid.active = compute.gpu_active
        assert_equal(grid.get_cell(cw, cz, cy, cx), 0)


fn test_gpu_wrap_across_faces_birth() raises:
    """Birth at origin using neighbors that wrap across opposite faces in 4D."""
    @parameter
    if has_accelerator():
        from hypercube.gpu_compute import GPUCompute
        
        var grid = Grid()
        var max_w = grid.w_dim - 1
        var max_x = grid.width - 1
        
        # Place six neighbors around (0,0,0,0) using wrap on axes
        grid.cells_a[grid._idx(0, 0, 0, 1)] = 1        # +x
        grid.cells_a[grid._idx(0, 0, 1, 0)] = 1        # +y
        grid.cells_a[grid._idx(0, 1, 0, 0)] = 1        # +z
        grid.cells_a[grid._idx(1, 0, 0, 0)] = 1        # +w
        grid.cells_a[grid._idx(0, 0, 0, max_x)] = 1    # -x via wrap
        grid.cells_a[grid._idx(max_w, 0, 0, 0)] = 1    # -w via wrap
        
        var compute = GPUCompute(
            grid.width,
            grid.height,
            grid.depth,
            grid.w_dim,
            grid.stride,
            grid.layer_stride,
            grid.hyperlayer_stride,
        )
        var src_ptr = grid.cells_a.unsafe_ptr()
        compute.upload_from_cpu(src_ptr, 0)
        compute.step()
        compute.download_to_cpu(grid.cells_b)
        grid.active = compute.gpu_active
        
        # Birth at origin should occur (B6 rule)
        assert_equal(grid.get_cell(0, 0, 0, 0), 1)


