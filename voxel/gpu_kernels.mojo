"""Native Mojo GPU kernels for the 3D voxel automaton (B6/S567).

The volume is toroidal in all three dimensions. Each thread updates one
voxel per generation using 26-neighbor counts.
"""

from gpu.host import DeviceContext
from gpu import block_dim, block_idx, thread_idx, barrier
from layout import Layout, LayoutTensor

from voxel.grid import VOXEL_WIDTH, VOXEL_HEIGHT, VOXEL_DEPTH

# GPU kernel constants
alias cell_dtype = DType.uint8
alias gpu_block_x = 8
alias gpu_block_y = 8
alias gpu_block_z = 4

# Compile-time layout (fixed grid dimensions)
alias GPU_STRIDE: Int = ((VOXEL_WIDTH + 63) // 64) * 64
alias GPU_LAYER_STRIDE: Int = GPU_STRIDE * VOXEL_HEIGHT
alias GRID_SIZE: Int = GPU_LAYER_STRIDE * VOXEL_DEPTH
alias grid_layout = Layout.row_major(GRID_SIZE)


@fieldwise_init
struct KernelDims:
    """GPU kernel launch dimensions."""
    var grid_x: Int
    var grid_y: Int
    var grid_z: Int
    var block_x: Int
    var block_y: Int
    var block_z: Int


fn get_kernel_dims(width: Int, height: Int, depth: Int) -> KernelDims:
    """Calculate kernel launch dimensions for a 3D grid."""
    var grid_x = (width + gpu_block_x - 1) // gpu_block_x
    var grid_y = (height + gpu_block_y - 1) // gpu_block_y
    var grid_z = (depth + gpu_block_z - 1) // gpu_block_z
    return KernelDims(grid_x, grid_y, grid_z, gpu_block_x, gpu_block_y, gpu_block_z)


# ─────────────────────────────────────────────────────────────────────────────
# Thread Indexing
# ─────────────────────────────────────────────────────────────────────────────


@always_inline
fn _get_thread_x() -> Int:
    """Get this thread's X coordinate (column)."""
    return Int(block_idx.x * block_dim.x + thread_idx.x)


@always_inline
fn _get_thread_y() -> Int:
    """Get this thread's Y coordinate (row)."""
    return Int(block_idx.y * block_dim.y + thread_idx.y)


@always_inline
fn _get_thread_z() -> Int:
    """Get this thread's Z coordinate (depth)."""
    return Int(block_idx.z * block_dim.z + thread_idx.z)


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate Wrapping (Toroidal Boundaries)
# ─────────────────────────────────────────────────────────────────────────────


@always_inline
fn _wrap(v: Int, limit: Int) -> Int:
    """Toroidal wrap helper."""
    if v < 0:
        return limit - 1
    elif v >= limit:
        return 0
    return v


# ─────────────────────────────────────────────────────────────────────────────
# Neighbor Counting
# ─────────────────────────────────────────────────────────────────────────────


@always_inline
fn _count_neighbors(
    grid: LayoutTensor[cell_dtype, grid_layout, MutAnyOrigin],
    x: Int,
    y: Int,
    z: Int,
    width: Int,
    height: Int,
    depth: Int,
    stride: Int,
    layer_stride: Int,
) -> Int:
    """Count 26 neighbors using wrapped coordinates."""
    var count = 0
    for dz in range(-1, 2):
        var zz = _wrap(z + dz, depth)
        var z_offset = zz * layer_stride
        for dy in range(-1, 2):
            var yy = _wrap(y + dy, height)
            var y_offset = z_offset + yy * stride
            for dx in range(-1, 2):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                var xx = _wrap(x + dx, width)
                count += Int(grid[y_offset + xx])
    return count


# ─────────────────────────────────────────────────────────────────────────────
# Rule (B6 / S567)
# ─────────────────────────────────────────────────────────────────────────────


@always_inline
fn _apply_voxel_rules(current: Int, neighbors: Int) -> UInt8:
    """Apply B6/S567: birth on 6, survive on 5/6/7."""
    if current == 1 and (neighbors == 5 or neighbors == 6 or neighbors == 7):
        return 1
    elif current == 0 and neighbors == 6:
        return 1
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# GPU Kernel
# ─────────────────────────────────────────────────────────────────────────────


fn voxel_generation_kernel(
    read_grid: LayoutTensor[cell_dtype, grid_layout, MutAnyOrigin],
    write_grid: LayoutTensor[cell_dtype, grid_layout, MutAnyOrigin],
    width: Int,
    height: Int,
    depth: Int,
    stride: Int,
    layer_stride: Int,
):
    """GPU kernel to compute one generation of the 3D automaton."""
    var x = _get_thread_x()
    var y = _get_thread_y()
    var z = _get_thread_z()
    
    if x >= width or y >= height or z >= depth:
        return
    
    var neighbors = _count_neighbors(
        read_grid, x, y, z, width, height, depth, stride, layer_stride
    )
    
    var idx = z * layer_stride + y * stride + x
    var current = Int(read_grid[idx])
    write_grid[idx] = _apply_voxel_rules(current, neighbors)


