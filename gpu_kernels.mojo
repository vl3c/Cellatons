"""Native Mojo GPU kernels for cellular automaton computation."""

from gpu.host import DeviceContext
from gpu import block_dim, block_idx, thread_idx
from layout import Layout, LayoutTensor
from common import WIDTH, HEIGHT

# GPU kernel constants
alias cell_dtype = DType.int32
alias gpu_block_size = 256

# Compile-time layouts for GPU tensors
alias row_layout = Layout.row_major(WIDTH)
alias grid_size = WIDTH * HEIGHT
alias grid_layout = Layout.row_major(grid_size)
alias max_patterns = 8  # Maximum number of patterns (8 possible for 3-bit patterns)
alias patterns_layout = Layout.row_major(max_patterns)


fn automaton_row_kernel(
    prev_row: LayoutTensor[cell_dtype, row_layout, MutAnyOrigin],
    curr_row: LayoutTensor[cell_dtype, row_layout, MutAnyOrigin],
    allowed_patterns: LayoutTensor[cell_dtype, patterns_layout, MutAnyOrigin],
    num_patterns: Int,
):
    """GPU kernel to compute one row of the cellular automaton (ping-pong version)."""
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    
    # Only process valid interior cells
    if tid >= 1 and tid < WIDTH - 1:
        var left = Int(prev_row[tid - 1])
        var center = Int(prev_row[tid])
        var right = Int(prev_row[tid + 1])
        var code = left * 4 + center * 2 + right
        
        # Check if code matches any allowed pattern
        var result: Int32 = 0
        for i in range(num_patterns):
            if Int(allowed_patterns[i]) == code:
                result = 1
                break
        curr_row[tid] = result


fn init_center_kernel(
    grid: LayoutTensor[cell_dtype, grid_layout, MutAnyOrigin],
):
    """GPU kernel to initialize the center cell of the first row."""
    grid[WIDTH // 2] = Int32(1)


fn automaton_grid_kernel(
    grid: LayoutTensor[cell_dtype, grid_layout, MutAnyOrigin],
    allowed_patterns: LayoutTensor[cell_dtype, patterns_layout, MutAnyOrigin],
    num_patterns: Int,
    row_idx: Int,
):
    """GPU kernel to compute one row of the cellular automaton (full grid version)."""
    var col = Int(block_idx.x * block_dim.x + thread_idx.x)
    
    # Only process valid interior cells
    if col >= 1 and col < WIDTH - 1:
        # Calculate offsets into flattened grid
        var prev_row_offset = (row_idx - 1) * WIDTH
        var curr_row_offset = row_idx * WIDTH
        
        var left = Int(grid[prev_row_offset + col - 1])
        var center = Int(grid[prev_row_offset + col])
        var right = Int(grid[prev_row_offset + col + 1])
        var code = left * 4 + center * 2 + right
        
        # Check if code matches any allowed pattern
        var result: Int32 = 0
        for i in range(num_patterns):
            if Int(allowed_patterns[i]) == code:
                result = 1
                break
        grid[curr_row_offset + col] = result

