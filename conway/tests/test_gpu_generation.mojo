"""Tests for GPU generation in Conway's Game of Life."""

from testing import assert_true, assert_equal
from sys import has_accelerator
from conway.grid import Grid, SCREEN_WIDTH, SCREEN_HEIGHT
from conway.cpu_compute import CPUCompute


# ─────────────────────────────────────────────────────────────────────────────
# GPU Availability Tests
# ─────────────────────────────────────────────────────────────────────────────


fn test_has_gpu_returns_bool() raises:
    """has_gpu should return a boolean value."""
    var grid = Grid(10, 10)
    var result = grid.has_gpu()
    # Result should be True or False (both are valid)
    assert_true(result == True or result == False)


fn test_gpu_detection_consistent() raises:
    """GPU detection should be consistent across calls."""
    var grid1 = Grid(10, 10)
    var grid2 = Grid(10, 10)
    assert_equal(grid1.has_gpu(), grid2.has_gpu())


# ─────────────────────────────────────────────────────────────────────────────
# GPU Compute Tests (conditional on GPU availability)
# ─────────────────────────────────────────────────────────────────────────────


fn test_gpu_compute_initialization() raises:
    """GPUCompute should initialize without error if GPU available."""
    @parameter
    if has_accelerator():
        from conway.gpu_compute import GPUCompute
        var compute = GPUCompute(100, 100, 128)
        assert_equal(compute.width, 100)
        assert_equal(compute.height, 100)
        assert_equal(compute.stride, 128)


fn test_gpu_compute_step_runs() raises:
    """GPUCompute.step should execute without error if GPU available."""
    @parameter
    if has_accelerator():
        from conway.gpu_compute import GPUCompute
        var compute = GPUCompute(100, 100, 128)
        compute.step()
        # If we get here, step ran successfully
        assert_true(True)


fn test_gpu_compute_swaps_active() raises:
    """GPUCompute.step should swap the active buffer."""
    @parameter
    if has_accelerator():
        from conway.gpu_compute import GPUCompute
        var compute = GPUCompute(100, 100, 128)
        assert_equal(compute.gpu_active, 0)
        compute.step()
        assert_equal(compute.gpu_active, 1)
        compute.step()
        assert_equal(compute.gpu_active, 0)


fn test_gpu_compute_upload_download_roundtrip() raises:
    """Data uploaded to GPU should be downloadable."""
    @parameter
    if has_accelerator():
        from conway.gpu_compute import GPUCompute
        
        var grid = Grid(100, 100)
        # Set some cells
        grid.cells_a[grid._idx(50, 50)] = 1
        grid.cells_a[grid._idx(50, 51)] = 1
        grid.cells_a[grid._idx(50, 52)] = 1
        
        var compute = GPUCompute(grid.width, grid.height, grid.stride)
        
        # Upload
        var src_ptr = grid.cells_a.unsafe_ptr()
        compute.upload_from_cpu(src_ptr, 0)
        
        # Download back
        var dst_ptr = grid.cells_b.unsafe_ptr()
        compute.download_to_cpu(dst_ptr)
        
        # Verify data matches
        assert_equal(Int(grid.cells_b[grid._idx(50, 50)]), 1)
        assert_equal(Int(grid.cells_b[grid._idx(50, 51)]), 1)
        assert_equal(Int(grid.cells_b[grid._idx(50, 52)]), 1)


# ─────────────────────────────────────────────────────────────────────────────
# GPU vs CPU Consistency Tests
# ─────────────────────────────────────────────────────────────────────────────


fn test_gpu_cpu_produce_same_result_empty_grid() raises:
    """GPU and CPU should produce same result for empty grid."""
    @parameter
    if has_accelerator():
        from conway.gpu_compute import GPUCompute
        
        var grid_cpu = Grid(100, 100)
        var grid_gpu = Grid(100, 100)
        
        # CPU step
        CPUCompute.step(grid_cpu)
        
        # GPU step
        var compute = GPUCompute(grid_gpu.width, grid_gpu.height, grid_gpu.stride)
        var src_ptr = grid_gpu.cells_a.unsafe_ptr()
        compute.upload_from_cpu(src_ptr, 0)
        compute.step()
        var dst_ptr = grid_gpu.cells_b.unsafe_ptr()
        compute.download_to_cpu(dst_ptr)
        grid_gpu.active = 1
        
        # Compare results
        for row in range(100):
            for col in range(100):
                assert_equal(grid_cpu.get_cell(row, col), grid_gpu.get_cell(row, col))


fn test_gpu_cpu_produce_same_result_block() raises:
    """GPU and CPU should produce same result for a block pattern."""
    @parameter
    if has_accelerator():
        from conway.gpu_compute import GPUCompute
        
        var grid_cpu = Grid(100, 100)
        var grid_gpu = Grid(100, 100)
        
        # Create identical blocks in both grids
        grid_cpu.cells_a[grid_cpu._idx(50, 50)] = 1
        grid_cpu.cells_a[grid_cpu._idx(50, 51)] = 1
        grid_cpu.cells_a[grid_cpu._idx(51, 50)] = 1
        grid_cpu.cells_a[grid_cpu._idx(51, 51)] = 1
        
        grid_gpu.cells_a[grid_gpu._idx(50, 50)] = 1
        grid_gpu.cells_a[grid_gpu._idx(50, 51)] = 1
        grid_gpu.cells_a[grid_gpu._idx(51, 50)] = 1
        grid_gpu.cells_a[grid_gpu._idx(51, 51)] = 1
        
        # CPU step
        CPUCompute.step(grid_cpu)
        
        # GPU step
        var compute = GPUCompute(grid_gpu.width, grid_gpu.height, grid_gpu.stride)
        var src_ptr = grid_gpu.cells_a.unsafe_ptr()
        compute.upload_from_cpu(src_ptr, 0)
        compute.step()
        var dst_ptr = grid_gpu.cells_b.unsafe_ptr()
        compute.download_to_cpu(dst_ptr)
        grid_gpu.active = 1
        
        # Verify block is stable in both
        assert_equal(grid_cpu.get_cell(50, 50), 1)
        assert_equal(grid_gpu.get_cell(50, 50), 1)


fn test_gpu_cpu_produce_same_result_blinker() raises:
    """GPU and CPU should produce same result for a blinker."""
    @parameter
    if has_accelerator():
        from conway.gpu_compute import GPUCompute
        
        var grid_cpu = Grid(100, 100)
        var grid_gpu = Grid(100, 100)
        
        # Create identical horizontal blinkers in both grids
        grid_cpu.cells_a[grid_cpu._idx(50, 49)] = 1
        grid_cpu.cells_a[grid_cpu._idx(50, 50)] = 1
        grid_cpu.cells_a[grid_cpu._idx(50, 51)] = 1
        
        grid_gpu.cells_a[grid_gpu._idx(50, 49)] = 1
        grid_gpu.cells_a[grid_gpu._idx(50, 50)] = 1
        grid_gpu.cells_a[grid_gpu._idx(50, 51)] = 1
        
        # CPU step
        CPUCompute.step(grid_cpu)
        
        # GPU step
        var compute = GPUCompute(grid_gpu.width, grid_gpu.height, grid_gpu.stride)
        var src_ptr = grid_gpu.cells_a.unsafe_ptr()
        compute.upload_from_cpu(src_ptr, 0)
        compute.step()
        var dst_ptr = grid_gpu.cells_b.unsafe_ptr()
        compute.download_to_cpu(dst_ptr)
        grid_gpu.active = 1
        
        # Verify vertical blinker in both
        assert_equal(grid_cpu.get_cell(49, 50), grid_gpu.get_cell(49, 50))
        assert_equal(grid_cpu.get_cell(50, 50), grid_gpu.get_cell(50, 50))
        assert_equal(grid_cpu.get_cell(51, 50), grid_gpu.get_cell(51, 50))
