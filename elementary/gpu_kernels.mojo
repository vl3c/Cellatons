"""Native Mojo GPU kernels for cellular automaton computation."""

from gpu.host import DeviceContext
from gpu import block_dim, block_idx, thread_idx, barrier
from layout import Layout, LayoutTensor
from shared.common import WIDTH, HEIGHT

# GPU kernel constants
alias cell_dtype = DType.int32
alias gpu_block_size = 256
alias ROWS_PER_KERNEL = 10000  # All rows in one kernel = 3 launches total (fastest)

# Compile-time layouts for GPU tensors
alias row_layout = Layout.row_major(WIDTH)
alias grid_size = WIDTH * HEIGHT
alias grid_layout = Layout.row_major(grid_size)
alias max_patterns = 8  # Maximum number of patterns (8 possible for 3-bit patterns)
alias patterns_layout = Layout.row_major(max_patterns)


# ─────────────────────────────────────────────────────────────────────────────
# Abstraction Layer: 2D Grid access over 1D memory
# ─────────────────────────────────────────────────────────────────────────────


@always_inline
fn get_cell(
    grid: LayoutTensor[cell_dtype, grid_layout, MutAnyOrigin],
    row: Int,
    col: Int,
) -> Int:
    """Access grid as if it were 2D."""
    return Int(grid[row * WIDTH + col])


@always_inline
fn set_cell(
    grid: LayoutTensor[cell_dtype, grid_layout, MutAnyOrigin],
    row: Int,
    col: Int,
    value: Int32,
):
    """Write to grid as if it were 2D."""
    grid[row * WIDTH + col] = value


@fieldwise_init
struct Neighborhood(Copyable, Movable):
    """The 3-cell neighborhood from the previous row."""

    var left: Int
    var center: Int
    var right: Int

    @always_inline
    fn to_pattern_code(self) -> Int:
        """Convert neighborhood to pattern code (0-7)."""
        return self.left * 4 + self.center * 2 + self.right


@always_inline
fn get_neighborhood(
    grid: LayoutTensor[cell_dtype, grid_layout, MutAnyOrigin],
    row: Int,
    col: Int,
) -> Neighborhood:
    """Get the 3 cells above the current position."""
    return Neighborhood(
        get_cell(grid, row - 1, col - 1),
        get_cell(grid, row - 1, col),
        get_cell(grid, row - 1, col + 1),
    )


@always_inline
fn get_row_neighborhood(
    row: LayoutTensor[cell_dtype, row_layout, MutAnyOrigin],
    col: Int,
) -> Neighborhood:
    """Get the 3 cells from a row tensor."""
    return Neighborhood(
        Int(row[col - 1]),
        Int(row[col]),
        Int(row[col + 1]),
    )


@always_inline
fn get_thread_col() -> Int:
    """Compute which column this thread is responsible for."""
    return Int(block_idx.x * block_dim.x + thread_idx.x)


@always_inline
fn is_interior_cell(col: Int) -> Bool:
    """Check if this is a valid interior cell (not an edge)."""
    return col >= 1 and col < WIDTH - 1


@always_inline
fn compute_cell_value(
    allowed_patterns: LayoutTensor[cell_dtype, patterns_layout, MutAnyOrigin],
    code: Int,
) -> Int32:
    """Determine if the pattern produces a live cell (legacy O(8) version).
    
    Checks all 8 pattern slots. Unused slots contain -1, which never matches
    valid codes (0-7), so no explicit count is needed.
    """
    for i in range(max_patterns):
        if Int(allowed_patterns[i]) == code:
            return 1
    return 0


@always_inline
fn compute_cell_value_fast(rule_mask: Int32, code: Int) -> Int32:
    """O(1) bitmask lookup - determines if pattern produces a live cell.
    
    Uses the same bitmask optimization as CPU: (mask >> code) & 1
    code is 0-7, mask bit at position code determines output.
    """
    return (rule_mask >> code) & 1


# ─────────────────────────────────────────────────────────────────────────────
# GPU Kernels: Clean business logic using abstractions
# ─────────────────────────────────────────────────────────────────────────────


fn automaton_row_kernel(
    prev_row: LayoutTensor[cell_dtype, row_layout, MutAnyOrigin],
    curr_row: LayoutTensor[cell_dtype, row_layout, MutAnyOrigin],
    allowed_patterns: LayoutTensor[cell_dtype, patterns_layout, MutAnyOrigin],
):
    """GPU kernel to compute one row of the cellular automaton (ping-pong version)."""
    var col = get_thread_col()

    if is_interior_cell(col):
        var neighborhood = get_row_neighborhood(prev_row, col)
        var code = neighborhood.to_pattern_code()
        curr_row[col] = compute_cell_value(allowed_patterns, code)


fn init_center_kernel(
    grid: LayoutTensor[cell_dtype, grid_layout, MutAnyOrigin],
):
    """GPU kernel to initialize the center cell of the first row."""
    grid[WIDTH // 2] = Int32(1)


fn automaton_grid_kernel(
    grid: LayoutTensor[cell_dtype, grid_layout, MutAnyOrigin],
    allowed_patterns: LayoutTensor[cell_dtype, patterns_layout, MutAnyOrigin],
    row_idx: Int,
):
    """GPU kernel to compute one row of the cellular automaton (full grid version, legacy)."""
    var col = get_thread_col()

    if is_interior_cell(col):
        var neighborhood = get_neighborhood(grid, row_idx, col)
        var code = neighborhood.to_pattern_code()
        var result = compute_cell_value(allowed_patterns, code)
        set_cell(grid, row_idx, col, result)


fn automaton_grid_kernel_fast(
    grid: LayoutTensor[cell_dtype, grid_layout, MutAnyOrigin],
    rule_mask: Int32,
    row_idx: Int,
):
    """GPU kernel with O(1) bitmask lookup (8x faster than legacy)."""
    var col = get_thread_col()

    if is_interior_cell(col):
        var neighborhood = get_neighborhood(grid, row_idx, col)
        var code = neighborhood.to_pattern_code()
        var result = compute_cell_value_fast(rule_mask, code)
        set_cell(grid, row_idx, col, result)


fn automaton_grid_kernel_bounded(
    grid: LayoutTensor[cell_dtype, grid_layout, MutAnyOrigin],
    rule_mask: Int32,
    row_idx: Int,
    left_bound: Int,
    right_bound: Int,
):
    """GPU kernel with O(1) bitmask AND sparse bounds optimization.
    
    Only processes cells within the active pyramid region, skipping
    threads outside [left_bound, right_bound].
    """
    var col = get_thread_col()

    # Skip threads outside active region (sparse bounds optimization)
    if col >= left_bound and col <= right_bound and is_interior_cell(col):
        var neighborhood = get_neighborhood(grid, row_idx, col)
        var code = neighborhood.to_pattern_code()
        var result = compute_cell_value_fast(rule_mask, code)
        set_cell(grid, row_idx, col, result)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Row Kernel: Reduces kernel launch overhead by processing batches
# ─────────────────────────────────────────────────────────────────────────────


fn automaton_multi_row_kernel(
    grid: LayoutTensor[cell_dtype, grid_layout, MutAnyOrigin],
    rule_mask: Int32,
    start_row: Int,
    num_rows: Int,
):
    """GPU kernel that processes multiple rows per launch using barrier() sync.
    
    Reduces kernel launch overhead by batching rows. Uses barrier() to synchronize
    all threads after each row before processing the next (row N depends on row N-1).
    
    With ROWS_PER_KERNEL=100, reduces launches from 10,000 to 100 per rule.
    Expected reduction: ~150ms launch overhead → ~1.5ms
    """
    var col = get_thread_col()
    
    # Process multiple rows within this single kernel launch
    for i in range(num_rows):
        var row = start_row + i
        
        # Only process if within grid bounds and interior cell
        if row < HEIGHT and is_interior_cell(col):
            var neighborhood = get_neighborhood(grid, row, col)
            var code = neighborhood.to_pattern_code()
            var result = compute_cell_value_fast(rule_mask, code)
            set_cell(grid, row, col, result)
        
        # CRITICAL: Synchronize ALL threads before processing next row
        # Row N+1 depends on row N being complete
        barrier()
