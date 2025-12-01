"""Native Mojo GPU kernels for cellular automaton computation."""

from gpu.host import DeviceContext
from gpu import block_dim, block_idx, thread_idx
from layout import Layout, LayoutTensor
from shared.common import WIDTH, HEIGHT

# GPU kernel constants
alias cell_dtype = DType.int32
alias gpu_block_size = 256

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
    """Determine if the pattern produces a live cell.
    
    Checks all 8 pattern slots. Unused slots contain -1, which never matches
    valid codes (0-7), so no explicit count is needed.
    """
    for i in range(max_patterns):
        if Int(allowed_patterns[i]) == code:
            return 1
    return 0


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
    """GPU kernel to compute one row of the cellular automaton (full grid version)."""
    var col = get_thread_col()

    if is_interior_cell(col):
        var neighborhood = get_neighborhood(grid, row_idx, col)
        var code = neighborhood.to_pattern_code()
        var result = compute_cell_value(allowed_patterns, code)
        set_cell(grid, row_idx, col, result)
