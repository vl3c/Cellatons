"""Conway's Game of Life grid with double buffering.

Pure data container for the 2D cell grid. Computation is handled by
separate modules (cpu_compute.mojo, gpu_compute.mojo).

All cells update simultaneously each generation using ping-pong buffers.
"""

from python import Python, PythonObject
from sys import has_accelerator
from memory import UnsafePointer

# Grid dimensions for 1440p fullscreen
alias SCREEN_WIDTH: Int = 2560
alias SCREEN_HEIGHT: Int = 1440
alias INITIAL_DENSITY: Float64 = 0.15

# SIMD configuration (AVX-512: 64 bytes = 64 cells)
alias simd_width = 64


struct Grid(Movable):
    """Double-buffered 2D grid for Conway's Game of Life.
    
    Uses ping-pong buffers: read from active, write to inactive, then swap.
    Supports toroidal wrapping (edges connect to opposite side).
    
    This is a pure data container - use CPUCompute or GPUCompute for generation.
    """
    var cells_a: List[UInt8]  # Buffer A
    var cells_b: List[UInt8]  # Buffer B
    var active: Int           # 0 = read A/write B, 1 = read B/write A
    var width: Int
    var height: Int
    var stride: Int           # Row stride aligned for SIMD
    
    # ─────────────────────────────────────────────────────────────────────────
    # Initialization
    # ─────────────────────────────────────────────────────────────────────────
    
    fn __init__(out self, width: Int, height: Int):
        """Initialize grid with given dimensions."""
        self.width = width
        self.height = height
        self.active = 0
        
        # Align stride to 64 bytes for AVX-512
        self.stride = ((width + simd_width - 1) // simd_width) * simd_width
        
        # Allocate both buffers
        var buffer_size = self.stride * height + simd_width  # Extra padding
        self.cells_a = List[UInt8](capacity=buffer_size)
        self.cells_b = List[UInt8](capacity=buffer_size)
        
        # Zero-initialize
        for _ in range(buffer_size):
            self.cells_a.append(0)
            self.cells_b.append(0)
    
    fn __moveinit__(out self, deinit existing: Self):
        """Move constructor."""
        self.cells_a = existing.cells_a^
        self.cells_b = existing.cells_b^
        self.active = existing.active
        self.width = existing.width
        self.height = existing.height
        self.stride = existing.stride
    
    fn __del__(deinit self):
        """Clean up resources."""
        pass
    
    # ─────────────────────────────────────────────────────────────────────────
    # Cell Access
    # ─────────────────────────────────────────────────────────────────────────
    
    @always_inline
    fn _idx(self, row: Int, col: Int) -> Int:
        """Convert 2D coordinates to 1D index."""
        return row * self.stride + col
    
    @always_inline
    fn get_cell(self, row: Int, col: Int) -> Int:
        """Get cell value from active buffer."""
        if self.active == 0:
            return Int(self.cells_a[self._idx(row, col)])
        else:
            return Int(self.cells_b[self._idx(row, col)])
    
    @always_inline
    fn set_cell(mut self, row: Int, col: Int, value: Int):
        """Set cell value in inactive buffer (for writing next generation)."""
        if self.active == 0:
            self.cells_b[self._idx(row, col)] = UInt8(value)
        else:
            self.cells_a[self._idx(row, col)] = UInt8(value)
    
    @always_inline
    fn get_cell_toroidal(self, row: Int, col: Int) -> Int:
        """Get cell with toroidal wrapping."""
        var r = row
        var c = col
        
        # Wrap row
        if r < 0:
            r = self.height - 1
        elif r >= self.height:
            r = 0
        
        # Wrap column
        if c < 0:
            c = self.width - 1
        elif c >= self.width:
            c = 0
        
        return self.get_cell(r, c)
    
    fn count_neighbors(self, row: Int, col: Int) -> Int:
        """Count live neighbors using toroidal wrapping (8-connected)."""
        var count = 0
        
        # All 8 neighbors
        count += self.get_cell_toroidal(row - 1, col - 1)
        count += self.get_cell_toroidal(row - 1, col)
        count += self.get_cell_toroidal(row - 1, col + 1)
        count += self.get_cell_toroidal(row, col - 1)
        count += self.get_cell_toroidal(row, col + 1)
        count += self.get_cell_toroidal(row + 1, col - 1)
        count += self.get_cell_toroidal(row + 1, col)
        count += self.get_cell_toroidal(row + 1, col + 1)
        
        return count
    
    # ─────────────────────────────────────────────────────────────────────────
    # Buffer Management
    # ─────────────────────────────────────────────────────────────────────────
    
    fn swap_buffers(mut self):
        """Swap active and inactive buffers."""
        self.active = 1 - self.active
    
    fn get_active_cells_ptr(mut self) -> Int:
        """Get integer pointer value to active buffer for rendering.
        
        Returns raw pointer value to avoid UnsafePointer origin issues.
        Caller should use this with ctypes in Python for numpy view.
        """
        if self.active == 0:
            return Int(self.cells_a.unsafe_ptr())
        else:
            return Int(self.cells_b.unsafe_ptr())
    
    # ─────────────────────────────────────────────────────────────────────────
    # Initialization Methods
    # ─────────────────────────────────────────────────────────────────────────
    
    fn randomize(mut self, density: Float64) raises:
        """Initialize grid with random cells at given density."""
        var random = Python.import_module("random")
        
        # Write to active buffer (will be read in first generation)
        for row in range(self.height):
            for col in range(self.width):
                var r = Float64(random.random())
                var value: UInt8 = 1 if r < density else 0
                if self.active == 0:
                    self.cells_a[self._idx(row, col)] = value
                else:
                    self.cells_b[self._idx(row, col)] = value
    
    # ─────────────────────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────────────────────
    
    fn has_gpu(self) -> Bool:
        """Check if GPU is available."""
        @parameter
        if has_accelerator():
            return True
        return False
