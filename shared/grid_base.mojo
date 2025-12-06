"""Shared helpers for ping-pong grid buffers (2D and 3D)."""

# Default SIMD width (bytes) for stride alignment.
alias DEFAULT_SIMD_WIDTH: Int = 64


@always_inline
fn calc_stride(width: Int, simd_width: Int = DEFAULT_SIMD_WIDTH) -> Int:
    """Compute SIMD-aligned stride."""
    return ((width + simd_width - 1) // simd_width) * simd_width


@always_inline
fn buffer_size_2d(stride: Int, height: Int, padding: Int = DEFAULT_SIMD_WIDTH) -> Int:
    """Total buffer size for 2D grid with optional padding."""
    return stride * height + padding


@always_inline
fn compute_layer_stride(stride: Int, height: Int) -> Int:
    """Compute layer stride for 3D grids."""
    return stride * height


@always_inline
fn buffer_size_3d(stride: Int, height: Int, depth: Int, padding: Int = DEFAULT_SIMD_WIDTH) -> Int:
    """Total buffer size for 3D grid with optional padding."""
    var layer_stride = compute_layer_stride(stride, height)
    return layer_stride * depth + padding


fn alloc_zeroed(size: Int) -> List[UInt8]:
    """Allocate zero-initialized buffer."""
    var buf = List[UInt8](capacity=size)
    for _ in range(size):
        buf.append(0)
    return buf^


@always_inline
fn swap_active(active: Int) -> Int:
    """Return toggled active buffer index."""
    return 1 - active


@always_inline
fn active_ptr(active: Int, cells_a: List[UInt8], cells_b: List[UInt8]) -> Int:
    """Return integer pointer to the active buffer."""
    if active == 0:
        return Int(cells_a.unsafe_ptr())
    return Int(cells_b.unsafe_ptr())

