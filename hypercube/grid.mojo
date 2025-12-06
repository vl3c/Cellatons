"""4D hypercube grid with double buffering (GPU-first).

Layout: W * Z * Y * X with X-stride aligned for SIMD and GPU coalescing.
Toroidal wrapping on all four axes.
"""

from sys import has_accelerator
from memory import UnsafePointer
from random import seed, random_float64
from shared.grid_base import (
    DEFAULT_SIMD_WIDTH,
    calc_stride,
    compute_layer_stride,
    alloc_zeroed,
    swap_active,
    active_ptr,
)

# Grid dimensions (fixed to match GPU kernels)
alias HYPER_WIDTH: Int = 32
alias HYPER_HEIGHT: Int = 32
alias HYPER_DEPTH: Int = 32
alias HYPER_W: Int = 32
alias INITIAL_DENSITY: Float64 = 0.15

# SIMD configuration (alignment)
alias simd_width = DEFAULT_SIMD_WIDTH


struct Grid(Movable):
    """Double-buffered 4D grid for the hypercube automaton."""
    var cells_a: List[UInt8]
    var cells_b: List[UInt8]
    var active: Int
    var width: Int
    var height: Int
    var depth: Int
    var w_dim: Int
    var stride: Int              # X stride (aligned)
    var layer_stride: Int        # YZ stride (stride * height)
    var hyperlayer_stride: Int   # W stride (layer_stride * depth)
    
    fn __init__(out self):
        """Initialize a fixed-size hypercube grid."""
        self.width = HYPER_WIDTH
        self.height = HYPER_HEIGHT
        self.depth = HYPER_DEPTH
        self.w_dim = HYPER_W
        self.active = 0
        
        self.stride = calc_stride(self.width, simd_width)
        self.layer_stride = compute_layer_stride(self.stride, self.height)
        self.hyperlayer_stride = self.layer_stride * self.depth
        
        var buffer_size = self.hyperlayer_stride * self.w_dim + simd_width
        self.cells_a = alloc_zeroed(buffer_size)
        self.cells_b = alloc_zeroed(buffer_size)
    
    fn __moveinit__(out self, deinit existing: Self):
        self.cells_a = existing.cells_a^
        self.cells_b = existing.cells_b^
        self.active = existing.active
        self.width = existing.width
        self.height = existing.height
        self.depth = existing.depth
        self.w_dim = existing.w_dim
        self.stride = existing.stride
        self.layer_stride = existing.layer_stride
        self.hyperlayer_stride = existing.hyperlayer_stride
    
    @always_inline
    fn _idx(self, w: Int, z: Int, y: Int, x: Int) -> Int:
        """Convert 4D coordinates to linear index."""
        return w * self.hyperlayer_stride + z * self.layer_stride + y * self.stride + x
    
    @always_inline
    fn get_cell(self, w: Int, z: Int, y: Int, x: Int) -> Int:
        """Read from active buffer."""
        var idx = self._idx(w, z, y, x)
        if self.active == 0:
            return Int(self.cells_a[idx])
        return Int(self.cells_b[idx])
    
    @always_inline
    fn set_cell(mut self, w: Int, z: Int, y: Int, x: Int, value: Int):
        """Write to inactive buffer."""
        var idx = self._idx(w, z, y, x)
        if self.active == 0:
            self.cells_b[idx] = UInt8(value)
        else:
            self.cells_a[idx] = UInt8(value)
    
    @always_inline
    fn _wrap(self, v: Int, limit: Int) -> Int:
        """Toroidal wrap for a single axis."""
        if v < 0:
            return limit - 1
        elif v >= limit:
            return 0
        return v
    
    fn get_cell_toroidal(self, w: Int, z: Int, y: Int, x: Int) -> Int:
        """Read with wraparound in all four dimensions."""
        var ww = self._wrap(w, self.w_dim)
        var zz = self._wrap(z, self.depth)
        var yy = self._wrap(y, self.height)
        var xx = self._wrap(x, self.width)
        return self.get_cell(ww, zz, yy, xx)
    
    fn count_neighbors(self, w: Int, z: Int, y: Int, x: Int) -> Int:
        """Count 80 neighbors in 3*3*3*3 neighborhood (excluding center)."""
        var count = 0
        for dw in range(-1, 2):
            for dz in range(-1, 2):
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if dw == 0 and dz == 0 and dy == 0 and dx == 0:
                            continue
                        count += self.get_cell_toroidal(w + dw, z + dz, y + dy, x + dx)
        return count
    
    fn swap_buffers(mut self):
        """Swap active and inactive buffers."""
        self.active = swap_active(self.active)
    
    fn get_active_cells_ptr(mut self) -> Int:
        """Get integer pointer value to active buffer for rendering."""
        return active_ptr(self.active, self.cells_a, self.cells_b)
    
    fn randomize(mut self, density: Float64) raises:
        """Initialize grid with random live cells using stdlib RNG."""
        seed()  # non-deterministic seed
        for w in range(self.w_dim):
            for z in range(self.depth):
                for y in range(self.height):
                    for x in range(self.width):
                        var r = random_float64(0.0, 1.0)
                        var value: UInt8 = 1 if r < density else 0
                        if self.active == 0:
                            self.cells_a[self._idx(w, z, y, x)] = value
                        else:
                            self.cells_b[self._idx(w, z, y, x)] = value
    
    fn has_gpu(self) -> Bool:
        """Check if GPU is available."""
        @parameter
        if has_accelerator():
            return True
        return False


