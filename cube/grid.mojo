"""3D cube grid with double buffering (GPU-first).

Dimensions are fixed for the GPU kernel. Stride is 64-byte aligned to
match SIMD-friendly layout, though CPU stepping is intentionally absent
for this GPU-only project.
"""

from python import Python
from sys import has_accelerator
from memory import UnsafePointer
from shared.grid_base import (
    DEFAULT_SIMD_WIDTH,
    active_ptr,
    alloc_zeroed,
    buffer_size_3d,
    calc_stride,
    compute_layer_stride,
    swap_active,
)

# Grid dimensions (cubic volume)
alias cube_WIDTH: Int = 96
alias cube_HEIGHT: Int = 96
alias cube_DEPTH: Int = 96
alias INITIAL_DENSITY: Float64 = 0.20

# SIMD configuration (alignment)
alias simd_width = DEFAULT_SIMD_WIDTH


struct Grid(Movable):
    """Double-buffered 3D grid for the cube automaton."""
    var cells_a: List[UInt8]
    var cells_b: List[UInt8]
    var active: Int
    var width: Int
    var height: Int
    var depth: Int
    var stride: Int          # X stride (aligned)
    var layer_stride: Int    # YZ stride (stride * height)
    
    fn __init__(out self):
        """Initialize a fixed-size cubic grid."""
        self.width = cube_WIDTH
        self.height = cube_HEIGHT
        self.depth = cube_DEPTH
        self.active = 0
        
        self.stride = calc_stride(self.width, simd_width)
        self.layer_stride = compute_layer_stride(self.stride, self.height)
        
        var buffer_size = buffer_size_3d(self.stride, self.height, self.depth, simd_width)
        self.cells_a = alloc_zeroed(buffer_size)
        self.cells_b = alloc_zeroed(buffer_size)
    
    fn __moveinit__(out self, deinit existing: Self):
        self.cells_a = existing.cells_a^
        self.cells_b = existing.cells_b^
        self.active = existing.active
        self.width = existing.width
        self.height = existing.height
        self.depth = existing.depth
        self.stride = existing.stride
        self.layer_stride = existing.layer_stride
    
    @always_inline
    fn _idx(self, z: Int, y: Int, x: Int) -> Int:
        """Convert 3D coordinates to linear index."""
        return z * self.layer_stride + y * self.stride + x
    
    @always_inline
    fn get_cell(self, z: Int, y: Int, x: Int) -> Int:
        """Read from active buffer."""
        var idx = self._idx(z, y, x)
        if self.active == 0:
            return Int(self.cells_a[idx])
        return Int(self.cells_b[idx])
    
    @always_inline
    fn set_cell(mut self, z: Int, y: Int, x: Int, value: Int):
        """Write to inactive buffer."""
        var idx = self._idx(z, y, x)
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
    
    fn get_cell_toroidal(self, z: Int, y: Int, x: Int) -> Int:
        """Read with wraparound in all three dimensions."""
        var zz = self._wrap(z, self.depth)
        var yy = self._wrap(y, self.height)
        var xx = self._wrap(x, self.width)
        return self.get_cell(zz, yy, xx)
    
    fn count_neighbors(self, z: Int, y: Int, x: Int) -> Int:
        """Count 26 neighbors in 3×3×3 neighborhood (excluding center)."""
        var count = 0
        for dz in range(-1, 2):
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    count += self.get_cell_toroidal(z + dz, y + dy, x + dx)
        return count
    
    fn swap_buffers(mut self):
        """Swap active and inactive buffers."""
        self.active = swap_active(self.active)
    
    fn get_active_cells_ptr(mut self) -> Int:
        """Get integer pointer value to active buffer for rendering."""
        return active_ptr(self.active, self.cells_a, self.cells_b)
    
    fn randomize(mut self, density: Float64) raises:
        """Initialize grid with random live cells."""
        var random = Python.import_module("random")
        for z in range(self.depth):
            for y in range(self.height):
                for x in range(self.width):
                    var r = Float64(random.random())
                    var value: UInt8 = 1 if r < density else 0
                    if self.active == 0:
                        self.cells_a[self._idx(z, y, x)] = value
                    else:
                        self.cells_b[self._idx(z, y, x)] = value
    
    fn has_gpu(self) -> Bool:
        """Check if GPU is available."""
        @parameter
        if has_accelerator():
            return True
        return False


