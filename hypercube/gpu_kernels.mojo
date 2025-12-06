"""Native Mojo GPU kernels for the 4D hypercube automaton (B6/S567).

The volume is toroidal across W, Z, Y, and X. Threads map a flattened
(W * Z) axis onto block_z to stay within 3D launch bounds.
"""

from gpu.host import DeviceContext
from gpu import block_dim, block_idx, thread_idx
from layout import Layout, LayoutTensor

from hypercube.grid import HYPER_WIDTH, HYPER_HEIGHT, HYPER_DEPTH, HYPER_W

# GPU kernel constants
alias cell_dtype = DType.uint8
alias gpu_block_x = 8
alias gpu_block_y = 4
alias gpu_block_z = 4  # covers flattened (W * Z)

# Compile-time layout (fixed grid dimensions)
alias GPU_STRIDE: Int = ((HYPER_WIDTH + 63) // 64) * 64
alias GPU_LAYER_STRIDE: Int = GPU_STRIDE * HYPER_HEIGHT
alias GPU_HYPERLAYER_STRIDE: Int = GPU_LAYER_STRIDE * HYPER_DEPTH
alias GPU_GRID_SIZE: Int = GPU_HYPERLAYER_STRIDE * HYPER_W
alias grid_layout = Layout.row_major(GPU_GRID_SIZE)


@fieldwise_init
struct KernelDims:
    """GPU kernel launch dimensions."""
    var grid_x: Int
    var grid_y: Int
    var grid_z: Int
    var block_x: Int
    var block_y: Int
    var block_z: Int


fn get_kernel_dims(width: Int, height: Int, depth: Int, w_dim: Int) -> KernelDims:
    """Calculate kernel launch dimensions for flattened (w*z) axis."""
    var combined_depth = depth * w_dim
    var grid_x = (width + gpu_block_x - 1) // gpu_block_x
    var grid_y = (height + gpu_block_y - 1) // gpu_block_y
    var grid_z = (combined_depth + gpu_block_z - 1) // gpu_block_z
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
fn _get_thread_z_flat() -> Int:
    """Get this thread's flattened Z coordinate (covers Z and W)."""
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
    w: Int,
    width: Int,
    height: Int,
    depth: Int,
    w_dim: Int,
    stride: Int,
    layer_stride: Int,
    hyperlayer_stride: Int,
) -> Int:
    """Count 80 neighbors using wrapped coordinates."""
    var count = 0
    for dw in range(-1, 2):
        var ww = _wrap(w + dw, w_dim)
        var w_offset = ww * hyperlayer_stride
        for dz in range(-1, 2):
            var zz = _wrap(z + dz, depth)
            var z_offset = w_offset + zz * layer_stride
            for dy in range(-1, 2):
                var yy = _wrap(y + dy, height)
                var y_offset = z_offset + yy * stride
                for dx in range(-1, 2):
                    if dx == 0 and dy == 0 and dz == 0 and dw == 0:
                        continue
                    var xx = _wrap(x + dx, width)
                    count += Int(grid[y_offset + xx])
    return count


# ─────────────────────────────────────────────────────────────────────────────
# Rule (B6 / S567)
# ─────────────────────────────────────────────────────────────────────────────


@always_inline
fn _apply_rules(current: Int, neighbors: Int) -> UInt8:
    """Apply B6/S567: birth on 6, survive on 5/6/7."""
    if current == 1 and (neighbors == 5 or neighbors == 6 or neighbors == 7):
        return 1
    elif current == 0 and neighbors == 6:
        return 1
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# GPU Kernel
# ─────────────────────────────────────────────────────────────────────────────


fn hypercube_generation_kernel(
    read_grid: LayoutTensor[cell_dtype, grid_layout, MutAnyOrigin],
    write_grid: LayoutTensor[cell_dtype, grid_layout, MutAnyOrigin],
    width: Int,
    height: Int,
    depth: Int,
    w_dim: Int,
    stride: Int,
    layer_stride: Int,
    hyperlayer_stride: Int,
):
    """GPU kernel to compute one generation of the 4D automaton."""
    var x = _get_thread_x()
    var y = _get_thread_y()
    var z_flat = _get_thread_z_flat()
    
    var combined_depth = depth * w_dim
    if x >= width or y >= height or z_flat >= combined_depth:
        return
    
    var w = z_flat // depth
    var z = z_flat - (w * depth)
    
    var neighbors = _count_neighbors(
        read_grid,
        x,
        y,
        z,
        w,
        width,
        height,
        depth,
        w_dim,
        stride,
        layer_stride,
        hyperlayer_stride,
    )
    
    var idx = w * hyperlayer_stride + z * layer_stride + y * stride + x
    var current = Int(read_grid[idx])
    write_grid[idx] = _apply_rules(current, neighbors)


