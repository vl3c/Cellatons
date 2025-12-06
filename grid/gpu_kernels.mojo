"""Native Mojo GPU kernels for Grid Game of Life.

Implements toroidal (wrap-around) boundary conditions.
Each thread processes one cell per generation.
Optimized with inlined coordinate wrapping and direct memory access.
"""

from gpu.host import DeviceContext
from gpu import block_dim, block_idx, thread_idx, barrier
from layout import Layout, LayoutTensor
from shared.display import DISPLAY_WIDTH, DISPLAY_HEIGHT

# GPU kernel constants
alias cell_dtype = DType.uint8
alias gpu_block_x = 16
alias gpu_block_y = 16

# Compile-time layout
alias grid_size = DISPLAY_WIDTH * DISPLAY_HEIGHT
alias grid_layout = Layout.row_major(grid_size)


@fieldwise_init
struct KernelDims:
    """GPU kernel launch dimensions."""
    var grid_x: Int
    var grid_y: Int
    var block_x: Int
    var block_y: Int


fn get_kernel_dims(width: Int, height: Int) -> KernelDims:
    """Calculate optimal kernel launch dimensions."""
    var grid_x = (width + gpu_block_x - 1) // gpu_block_x
    var grid_y = (height + gpu_block_y - 1) // gpu_block_y
    return KernelDims(grid_x, grid_y, gpu_block_x, gpu_block_y)


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


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate Wrapping (Toroidal Boundaries)
# ─────────────────────────────────────────────────────────────────────────────


@always_inline
fn _wrap_left(x: Int, width: Int) -> Int:
    """Wrap x-1 for toroidal left boundary."""
    return x - 1 if x > 0 else width - 1


@always_inline
fn _wrap_right(x: Int, width: Int) -> Int:
    """Wrap x+1 for toroidal right boundary."""
    return x + 1 if x < width - 1 else 0


@always_inline
fn _wrap_above(y: Int, height: Int) -> Int:
    """Wrap y-1 for toroidal top boundary."""
    return y - 1 if y > 0 else height - 1


@always_inline
fn _wrap_below(y: Int, height: Int) -> Int:
    """Wrap y+1 for toroidal bottom boundary."""
    return y + 1 if y < height - 1 else 0


# ─────────────────────────────────────────────────────────────────────────────
# Neighbor Counting
# ─────────────────────────────────────────────────────────────────────────────


@always_inline
fn _count_neighbors(
    grid: LayoutTensor[cell_dtype, grid_layout, MutAnyOrigin],
    x: Int,
    x_left: Int,
    x_right: Int,
    offset_above: Int,
    offset_curr: Int,
    offset_below: Int,
) -> Int:
    """Count all 8 neighbors using pre-computed offsets."""
    var count = Int(grid[offset_above + x_left])    # top-left
    count += Int(grid[offset_above + x])            # top
    count += Int(grid[offset_above + x_right])      # top-right
    count += Int(grid[offset_curr + x_left])        # left
    count += Int(grid[offset_curr + x_right])       # right
    count += Int(grid[offset_below + x_left])       # bottom-left
    count += Int(grid[offset_below + x])            # bottom
    count += Int(grid[offset_below + x_right])      # bottom-right
    return count


# ─────────────────────────────────────────────────────────────────────────────
# Grid rules
# ─────────────────────────────────────────────────────────────────────────────


@always_inline
fn _apply_grid_rules(current: Int, neighbors: Int) -> UInt8:
    """Apply Grid Game of Life rules.
    
    Returns 1 if cell should be alive, 0 otherwise.
    - Live cell with 2-3 neighbors survives
    - Dead cell with exactly 3 neighbors becomes alive
    - All other cells die or stay dead
    """
    if current == 1 and (neighbors == 2 or neighbors == 3):
        return 1
    elif current == 0 and neighbors == 3:
        return 1
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# GPU Kernel
# ─────────────────────────────────────────────────────────────────────────────


fn grid_generation_kernel(
    read_grid: LayoutTensor[cell_dtype, grid_layout, MutAnyOrigin],
    write_grid: LayoutTensor[cell_dtype, grid_layout, MutAnyOrigin],
    width: Int,
    height: Int,
    stride: Int,
):
    """GPU kernel to compute one generation of Grid Game of Life.
    
    Each thread processes one cell. Uses toroidal boundary conditions.
    """
    # Get thread coordinates
    var x = _get_thread_x()
    var y = _get_thread_y()
    
    # Bounds check
    if x >= width or y >= height:
        return
    
    # Compute wrapped neighbor coordinates
    var x_left = _wrap_left(x, width)
    var x_right = _wrap_right(x, width)
    var y_above = _wrap_above(y, height)
    var y_below = _wrap_below(y, height)
    
    # Pre-compute row offsets
    var offset_above = y_above * stride
    var offset_curr = y * stride
    var offset_below = y_below * stride
    
    # Count neighbors
    var neighbors = _count_neighbors(
        read_grid, x, x_left, x_right,
        offset_above, offset_curr, offset_below
    )
    
    # Get current cell and apply rules
    var current = Int(read_grid[offset_curr + x])
    var result = _apply_grid_rules(current, neighbors)
    
    # Write result
    write_grid[offset_curr + x] = result
