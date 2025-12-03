"""CPU computation for Conway's Game of Life.

Implements SIMD-accelerated, parallel row processing for CPU generation.
Uses vectorized neighbor counting for interior cells and scalar operations
for edge cells that require toroidal wrapping.
"""

from algorithm import parallelize
from memory import UnsafePointer
from conway.rules import apply_conway_rules
from conway.grid import Grid

# SIMD configuration (AVX-512: 64 bytes = 64 cells)
alias simd_width = 64


struct CPUCompute:
    """CPU compute handler for Conway's Game of Life.
    
    Provides SIMD-accelerated generation computation using parallel row processing.
    """
    
    # ─────────────────────────────────────────────────────────────────────────
    # Helper Functions
    # ─────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    @always_inline
    fn _count_neighbors_at(
        read_ptr: UnsafePointer[UInt8],
        above_offset: Int,
        curr_offset: Int,
        below_offset: Int,
        col: Int,
        col_left: Int,
        col_right: Int,
    ) -> Int:
        """Count 8 neighbors at given position using pre-computed offsets."""
        var neighbors = Int(read_ptr[above_offset + col_left])
        neighbors += Int(read_ptr[above_offset + col])
        neighbors += Int(read_ptr[above_offset + col_right])
        neighbors += Int(read_ptr[curr_offset + col_left])
        neighbors += Int(read_ptr[curr_offset + col_right])
        neighbors += Int(read_ptr[below_offset + col_left])
        neighbors += Int(read_ptr[below_offset + col])
        neighbors += Int(read_ptr[below_offset + col_right])
        return neighbors
    
    @staticmethod
    @always_inline
    fn _load_neighbor_sum_simd(
        read_ptr: UnsafePointer[UInt8],
        above_offset: Int,
        curr_offset: Int,
        below_offset: Int,
        col: Int,
    ) -> SIMD[DType.uint8, simd_width]:
        """Load and sum all 8 neighbors using SIMD operations."""
        alias sw = simd_width
        
        # Load all 8 neighbor directions
        var above_left = (read_ptr + above_offset + col - 1).load[width=sw]()
        var above_center = (read_ptr + above_offset + col).load[width=sw]()
        var above_right = (read_ptr + above_offset + col + 1).load[width=sw]()
        var curr_left = (read_ptr + curr_offset + col - 1).load[width=sw]()
        var curr_right = (read_ptr + curr_offset + col + 1).load[width=sw]()
        var below_left = (read_ptr + below_offset + col - 1).load[width=sw]()
        var below_center = (read_ptr + below_offset + col).load[width=sw]()
        var below_right = (read_ptr + below_offset + col + 1).load[width=sw]()
        
        return (above_left + above_center + above_right + 
                curr_left + curr_right +
                below_left + below_center + below_right)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Main Computation
    # ─────────────────────────────────────────────────────────────────────────
    
    @staticmethod
    fn step(mut grid: Grid):
        """Compute one generation using SIMD + parallel rows.
        
        Uses vectorized neighbor counting for interior cells (not edges).
        Edge cells are processed with scalar operations for toroidal wrapping.
        
        Args:
            grid: The grid to compute the next generation for. Modified in-place.
        """
        var height = grid.height
        var width = grid.width
        var stride = grid.stride
        
        var ptr_a = grid.cells_a.unsafe_ptr()
        var ptr_b = grid.cells_b.unsafe_ptr()
        var active = grid.active
        
        @parameter
        fn process_row(row: Int):
            var read_ptr = ptr_a if active == 0 else ptr_b
            var write_ptr = ptr_b if active == 0 else ptr_a
            
            # Pre-compute row offsets with toroidal wrapping
            var row_above = (row - 1 + height) % height
            var row_below = (row + 1) % height
            
            var above_offset = row_above * stride
            var curr_offset = row * stride
            var below_offset = row_below * stride
            
            # ─────────────────────────────────────────────────────────────────
            # Left edge cell (col=0) - wraps to right edge
            # ─────────────────────────────────────────────────────────────────
            var neighbors = CPUCompute._count_neighbors_at(
                read_ptr, above_offset, curr_offset, below_offset,
                col=0, col_left=width - 1, col_right=1
            )
            var current = Int(read_ptr[curr_offset])
            write_ptr[curr_offset] = apply_conway_rules(current, neighbors)
            
            # ─────────────────────────────────────────────────────────────────
            # Interior cells using SIMD (col 1 to width-2)
            # ─────────────────────────────────────────────────────────────────
            alias sw = simd_width
            var col = 1
            while col + sw <= width - 1:
                var center = (read_ptr + curr_offset + col).load[width=sw]()
                var neighbor_sum = CPUCompute._load_neighbor_sum_simd(
                    read_ptr, above_offset, curr_offset, below_offset, col
                )
                
                # Apply Conway rules per-element
                for i in range(sw):
                    write_ptr[curr_offset + col + i] = apply_conway_rules(
                        Int(center[i]), Int(neighbor_sum[i])
                    )
                col += sw
            
            # ─────────────────────────────────────────────────────────────────
            # Remaining interior cells (scalar)
            # ─────────────────────────────────────────────────────────────────
            while col < width - 1:
                neighbors = CPUCompute._count_neighbors_at(
                    read_ptr, above_offset, curr_offset, below_offset,
                    col, col - 1, col + 1
                )
                current = Int(read_ptr[curr_offset + col])
                write_ptr[curr_offset + col] = apply_conway_rules(current, neighbors)
                col += 1
            
            # ─────────────────────────────────────────────────────────────────
            # Right edge cell (col=width-1) - wraps to left edge
            # ─────────────────────────────────────────────────────────────────
            neighbors = CPUCompute._count_neighbors_at(
                read_ptr, above_offset, curr_offset, below_offset,
                col=width - 1, col_left=width - 2, col_right=0
            )
            current = Int(read_ptr[curr_offset + width - 1])
            write_ptr[curr_offset + width - 1] = apply_conway_rules(current, neighbors)
        
        parallelize[process_row](height)
        grid.swap_buffers()

