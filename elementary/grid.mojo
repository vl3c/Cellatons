from python import Python, PythonObject
from elementary.rule import Rule
from algorithm import parallelize, vectorize
from sys.info import has_nvidia_gpu_accelerator, simdwidthof
from sys import has_accelerator
from shared.common import PIXELS_PER_CELL, WIDTH, HEIGHT
from shared.gpu_timing_result import GPUTimingResult
from shared.logger import Logger
from math import ceildiv

# Native GPU imports
from gpu.host import DeviceContext, HostBuffer
from layout import LayoutTensor
from elementary.gpu_kernels import (
    cell_dtype,
    gpu_block_size,
    grid_size,
    grid_layout,
    row_layout,
    max_patterns,
    patterns_layout,
    init_center_kernel,
    automaton_grid_kernel,
    automaton_row_kernel,
)

# SIMD configuration
alias simd_width = 64  # AVX-512: 512 bits / 8 bits = 64 cells


struct Grid(Copyable, Movable):
    var cells: List[UInt8]  # Flat buffer with 2D indexing: cells[row * width + col]
    var width: Int
    var height: Int
    var logger: Logger
    
    fn __init__(out self, width: Int, height: Int, logger: Logger):
        self.logger = logger.copy()
        self.logger.log("Grid.__init__ starting")
        self.logger.log_int("  width:", width)
        self.logger.log_int("  height:", height)
        self.width = width
        self.height = height
        
        # Allocate flat buffer with padding for SIMD safety
        var flat_size = width * height + simd_width
        self.cells = List[UInt8](capacity=flat_size)
        # Zero-initialize
        for _ in range(flat_size):
            self.cells.append(0)
        
        self.logger.log("Grid.__init__ complete")
    
    fn __copyinit__(out self, existing: Self):
        self.width = existing.width
        self.height = existing.height
        self.logger = existing.logger.copy()
        
        # Copy flat buffer
        self.cells = List[UInt8](capacity=len(existing.cells))
        for i in range(len(existing.cells)):
            self.cells.append(existing.cells[i])
    
    fn __moveinit__(out self, deinit existing: Self):
        self.width = existing.width
        self.height = existing.height
        self.logger = existing.logger^
        self.cells = existing.cells^
    
    @always_inline
    fn set_cell(mut self, row: Int, col: Int, value: Int):
        self.cells[row * self.width + col] = UInt8(value)
    
    @always_inline
    fn get_cell(self, row: Int, col: Int) -> Int:
        return Int(self.cells[row * self.width + col])
    
    @always_inline
    fn _idx(self, row: Int, col: Int) -> Int:
        """Fast 2D to 1D index conversion."""
        return row * self.width + col
    
    fn get_width(self) -> Int:
        return self.width
    
    fn get_height(self) -> Int:
        return self.height
    
    fn generate_parallel_cpu(mut self, rule: Rule):
        self.logger.log("Entering generate_parallel_cpu")
        var center = self.width // 2
        self.cells[center] = 1  # Row 0, center column
        
        var left_bound = center
        var right_bound = center
        
        for row in range(1, self.height):
            # Expand bounds by 1 on each side, clamped to grid edges
            left_bound = left_bound - 1 if left_bound > 1 else 1
            right_bound = right_bound + 1 if right_bound < self.width - 2 else self.width - 2
            var bound_width = right_bound - left_bound + 1
            var lb = left_bound  # Capture for closure
            var prev_offset = (row - 1) * self.width
            var curr_offset = row * self.width
            
            @parameter
            fn compute_cell(offset: Int):
                var col = lb + offset
                var left = Int(self.cells[prev_offset + col - 1])
                var center_val = Int(self.cells[prev_offset + col])
                var right = Int(self.cells[prev_offset + col + 1])
                self.cells[curr_offset + col] = UInt8(rule.apply(left, center_val, right))
            
            parallelize[compute_cell](bound_width)
        self.logger.log("Completed generate_parallel_cpu")
    
    fn generate_simd_cpu(mut self, rule: Rule):
        """Generate using SIMD with sparse bounds optimization."""
        self.logger.log("Entering generate_simd_cpu")
        var center = self.width // 2
        
        # Initialize center cell
        self.cells[center] = 1
        
        # Pre-broadcast mask for SIMD operations
        var mask = rule.pattern_mask
        
        var left_bound = center
        var right_bound = center
        
        for row in range(1, self.height):
            # Expand bounds (inverted pyramid)
            left_bound = max(1, left_bound - 1)
            right_bound = min(self.width - 2, right_bound + 1)
            
            self._apply_rule_simd_row_fast(row, mask, left_bound, right_bound)
        
        self.logger.log("Completed generate_simd_cpu")
    
    fn _apply_rule_simd_row_fast(mut self, row: Int, mask: Int, left: Int, right: Int):
        """Process row using fully vectorized SIMD operations with 4x loop unrolling."""
        var prev_offset = (row - 1) * self.width
        var curr_offset = row * self.width
        var bound_width = right - left + 1
        var ptr = self.cells.unsafe_ptr()
        
        # Broadcast mask to SIMD vector once
        var mask_vec = SIMD[DType.int32, simd_width](mask)
        
        @always_inline
        @parameter
        fn process_chunk(col: Int):
            """Process a single SIMD chunk of 64 cells."""
            # Load 3 neighbors from previous row
            var l = (ptr + prev_offset + col - 1).load[width=simd_width]()
            var c = (ptr + prev_offset + col).load[width=simd_width]()
            var r = (ptr + prev_offset + col + 1).load[width=simd_width]()
            
            # Compute pattern codes: left*4 + center*2 + right
            var codes = l.cast[DType.int32]() * 4 + c.cast[DType.int32]() * 2 + r.cast[DType.int32]()
            
            # Parallel bitmask lookup: (mask >> codes) & 1 for all 64 values at once
            var results = ((mask_vec >> codes) & 1).cast[DType.uint8]()
            
            # Store results to current row
            (ptr + curr_offset + col).store(results)
        
        # Calculate chunks
        var full_chunks = bound_width // simd_width
        var unrolled_iterations = full_chunks // 4
        
        # 4x unrolled loop - processes 256 cells per iteration
        # Reduces loop overhead and enables better instruction pipelining
        for i in range(unrolled_iterations):
            var base_col = left + i * 4 * simd_width
            process_chunk(base_col)
            process_chunk(base_col + simd_width)
            process_chunk(base_col + 2 * simd_width)
            process_chunk(base_col + 3 * simd_width)
        
        # Handle remaining full chunks (0-3 chunks)
        var remaining_chunk_start = unrolled_iterations * 4
        for chunk in range(remaining_chunk_start, full_chunks):
            process_chunk(left + chunk * simd_width)
        
        # Process remaining cells with scalar bitmask (still O(1) per cell)
        var remaining_start = left + full_chunks * simd_width
        for col in range(remaining_start, right + 1):
            var l = Int(self.cells[prev_offset + col - 1])
            var c = Int(self.cells[prev_offset + col])
            var r = Int(self.cells[prev_offset + col + 1])
            var code = l * 4 + c * 2 + r
            self.cells[curr_offset + col] = UInt8((mask >> code) & 1)
    
    fn generate_sequential_cpu(mut self, rule: Rule):
        self.logger.log("Entering generate_sequential_cpu")
        var center = self.width // 2
        self.cells[center] = 1  # Row 0, center column
        
        var left_bound = center
        var right_bound = center
        
        for row in range(1, self.height):
            # Expand bounds by 1 on each side, clamped to grid edges
            left_bound = left_bound - 1 if left_bound > 1 else 1
            right_bound = right_bound + 1 if right_bound < self.width - 2 else self.width - 2
            self._apply_rule_cpu_row_bounded(row, rule, left_bound, right_bound)
        self.logger.log("Completed generate_sequential_cpu")

    fn generate_parallel_cells_cupy_gpu(mut self, rule: Rule) raises -> GPUTimingResult:
        self.logger.log("Entering generate_parallel_cells_cupy_gpu")
        # Set initial cell in the middle of the first row
        self.cells[self.width // 2] = 1  # Row 0, center column
        
        var py_time = Python.import_module("time")
        var py_builder = Python.import_module("builtins")
        var py_operator = Python.import_module("operator")
        var prep_duration: Float64 = 0.0
        var compute_duration: Float64 = 0.0
        var transfer_duration: Float64 = 0.0
        var total_duration: Float64 = 0.0
        var runs = 0
        # Check if NVIDIA GPU is available
        if has_nvidia_gpu_accelerator():
            self.logger.log("NVIDIA GPU detected, using CuPy")
            var cp = Python.import_module("cupy")
            
            var prep_start = py_time.time()
            self.logger.log("Initializing CuPy GPU grid")
            var grid_gpu = self._init_gpu_grid(cp, py_builder, py_operator)
            self.logger.log("Creating allowed patterns array")
            var allowed_array = self._create_allowed_patterns_array(rule, cp, py_builder)
            var prep_end = py_time.time()
            prep_duration = Float64(py_operator.sub(prep_end, prep_start))
            
            var compute_start = py_time.time()
            self.logger.log("Starting CuPy row computation")
            self._compute_rows_on_gpu(grid_gpu, allowed_array, cp, py_operator)
            var compute_end = py_time.time()
            compute_duration = Float64(py_operator.sub(compute_end, compute_start))
            self.logger.log("CuPy computation complete")
            
            var total_end = py_time.time()
            total_duration = Float64(py_operator.sub(total_end, prep_start))
            runs = 1
        else:
            self.logger.log("No NVIDIA GPU detected, falling back to CPU")
            print("No NVIDIA GPU detected, using CPU fallback")
            # CPU fallback
            for row in range(1, self.height):
                self._apply_rule_cpu_row(row, rule)
        
        return GPUTimingResult(prep_duration, compute_duration, transfer_duration, total_duration, runs)

    fn generate_native_gpu(mut self, rule: Rule) raises -> GPUTimingResult:
        """Generate cellular automaton using native Mojo GPU.
        
        Attempts full-grid mode first (faster). If GPU memory allocation fails,
        automatically falls back to ping-pong mode (slower but memory-efficient).
        """
        var py_time = Python.import_module("time")
        var py_builtins = Python.import_module("builtins")
        var py_operator = Python.import_module("operator")
        
        @parameter
        if not has_accelerator():
            self.logger.log("No GPU accelerator detected")
            print("No GPU accelerator, using CPU fallback")
            self.generate_sequential_cpu(rule)
            return GPUTimingResult(0.0, 0.0, 0.0, 0.0, 0)
        
        self.logger.log("GPU accelerator available, attempting full-grid mode")
        
        # Try full-grid mode first
        try:
            self.logger.log("Entering full-grid GPU path")
            return self._generate_native_gpu_full_grid(
                rule, py_time, py_builtins, py_operator
            )
        except:
            # Full grid allocation failed, fall back to ping-pong mode
            self.logger.log("Full-grid allocation failed, falling back to ping-pong mode")
            print("GPU memory insufficient for full grid, using ping-pong mode")
            return self._generate_native_gpu_pingpong(
                rule, py_time, py_builtins, py_operator
            )

    fn _generate_native_gpu_full_grid(
        mut self,
        rule: Rule,
        py_time: PythonObject,
        py_builtins: PythonObject,
        py_operator: PythonObject,
    ) raises -> GPUTimingResult:
        """Generate cellular automaton using full-grid GPU mode (fast).
        
        Allocates the entire grid on GPU, processes all rows, then optionally
        transfers back to CPU. Fastest mode but requires grid_size * 4 bytes of VRAM.
        """
        var transfer_duration: Float64 = 0.0  # No transfer in full-grid benchmark mode
        
        var prep_start = py_time.time()
        
        # Build allowed patterns
        var allowed = Self._build_allowed_patterns_static(rule)
        self.logger.log_int("Built allowed patterns, count:", len(allowed))
        
        # Create device context
        self.logger.log("Creating GPU device context")
        var ctx = DeviceContext()
        
        # Allocate all buffers (host + device)
        # This may throw if GPU memory is insufficient
        self.logger.log_int("Allocating host patterns buffer, size:", max_patterns)
        var host_patterns = ctx.enqueue_create_host_buffer[cell_dtype](max_patterns)
        self.logger.log_int("Allocating device grid buffer, size:", grid_size)
        var dev_grid = ctx.enqueue_create_buffer[cell_dtype](grid_size)
        self.logger.log_int("Allocating device patterns buffer, size:", max_patterns)
        var dev_patterns = ctx.enqueue_create_buffer[cell_dtype](max_patterns)
        self.logger.log("Synchronizing allocations")
        ctx.synchronize()  # Wait for all allocations
        self.logger.log("All GPU allocations successful")
        
        # Initialize patterns on host (pad with -1 for unused slots)
        Self._init_patterns_buffer(host_patterns, allowed)
        
        # Copy patterns to device
        ctx.enqueue_copy(dev_patterns, host_patterns)
        ctx.synchronize()  # Wait for copy before kernel launch
        
        # Create tensor view for the grid
        var grid_tensor = LayoutTensor[cell_dtype, grid_layout](dev_grid)
        
        var prep_end = py_time.time()
        var prep_duration = Float64(py_operator.sub(prep_end, prep_start))
        
        var compute_start = py_time.time()
        
        # Calculate grid dimensions for kernel launch
        var num_blocks = Self._get_num_blocks()
        var patterns_tensor = LayoutTensor[cell_dtype, patterns_layout](dev_patterns)
        
        # First, set the initial cell (run init kernel with 1 thread)
        ctx.enqueue_function_checked[init_center_kernel, init_center_kernel](
            grid_tensor,
            grid_dim=1,
            block_dim=1,
        )
        
        # Process ALL rows without syncing - just enqueue all kernels
        for row in range(1, HEIGHT):
            ctx.enqueue_function_checked[automaton_grid_kernel, automaton_grid_kernel](
                grid_tensor,
                patterns_tensor,
                row,
                grid_dim=num_blocks,
                block_dim=gpu_block_size,
            )
        
        # Single sync after all kernels are enqueued
        ctx.synchronize()
        
        var compute_end = py_time.time()
        var compute_duration = Float64(py_operator.sub(compute_end, compute_start))
        
        # Note: We skip GPU→CPU transfer since we use CPU results for rendering
        # The grid stays on GPU and is discarded (benchmark only)
        
        var total_end = py_time.time()
        var total_duration = Float64(py_operator.sub(total_end, prep_start))
        
        return GPUTimingResult(prep_duration, compute_duration, transfer_duration, total_duration, 1)

    fn _generate_native_gpu_pingpong(
        mut self,
        rule: Rule,
        py_time: PythonObject,
        py_builtins: PythonObject,
        py_operator: PythonObject,
    ) raises -> GPUTimingResult:
        """Generate cellular automaton using ping-pong GPU mode (memory-efficient).
        
        Uses only 2 row buffers on GPU, transferring each computed row back to CPU.
        Slower due to per-row transfers, but works when full grid doesn't fit in VRAM.
        """
        self.logger.log("Entering ping-pong GPU mode")
        var prep_start = py_time.time()
        
        # Build allowed patterns
        var allowed = Self._build_allowed_patterns_static(rule)
        self.logger.log_int("Built allowed patterns, count:", len(allowed))
        
        # Create device context
        self.logger.log("Creating GPU device context for ping-pong")
        var ctx = DeviceContext()
        
        # Allocate ping-pong row buffers on GPU (only 2 rows!)
        self.logger.log_int("Allocating ping-pong row buffer A, size:", WIDTH)
        var dev_row_a = ctx.enqueue_create_buffer[cell_dtype](WIDTH)
        self.logger.log_int("Allocating ping-pong row buffer B, size:", WIDTH)
        var dev_row_b = ctx.enqueue_create_buffer[cell_dtype](WIDTH)
        self.logger.log_int("Allocating device patterns buffer, size:", max_patterns)
        var dev_patterns = ctx.enqueue_create_buffer[cell_dtype](max_patterns)
        
        # Host buffers for initialization and transfer
        self.logger.log("Allocating host buffers for transfer")
        var host_row = ctx.enqueue_create_host_buffer[cell_dtype](WIDTH)
        var host_patterns = ctx.enqueue_create_host_buffer[cell_dtype](max_patterns)
        self.logger.log("Synchronizing allocations")
        ctx.synchronize()
        self.logger.log("Ping-pong allocations successful")
        
        # Initialize patterns on host (pad with -1 for unused slots)
        Self._init_patterns_buffer(host_patterns, allowed)
        
        # Initialize first row on host (center cell = 1)
        for i in range(WIDTH):
            host_row[i] = Int32(0)
        host_row[WIDTH // 2] = Int32(1)
        
        # Copy to device
        ctx.enqueue_copy(dev_patterns, host_patterns)
        ctx.enqueue_copy(dev_row_a, host_row)
        ctx.synchronize()
        
        # Store first row in CPU grid
        self.cells[WIDTH // 2] = 1  # Row 0, center column
        
        # Create tensor views
        var row_a_tensor = LayoutTensor[cell_dtype, row_layout](dev_row_a)
        var row_b_tensor = LayoutTensor[cell_dtype, row_layout](dev_row_b)
        var patterns_tensor = LayoutTensor[cell_dtype, patterns_layout](dev_patterns)
        
        var prep_end = py_time.time()
        var prep_duration = Float64(py_operator.sub(prep_end, prep_start))
        
        var compute_start = py_time.time()
        
        # Calculate kernel launch dimensions
        var num_blocks = Self._get_num_blocks()
        
        # Process rows with ping-pong pattern
        self.logger.log_int("Starting ping-pong processing, total rows:", HEIGHT - 1)
        for row in range(1, HEIGHT):
            # Log first few iterations for debugging
            if row <= 3:
                self.logger.log_int("Processing ping-pong row:", row)
            if row % 2 == 1:
                # Odd rows: A -> B
                ctx.enqueue_function_checked[automaton_row_kernel, automaton_row_kernel](
                    row_a_tensor,
                    row_b_tensor,
                    patterns_tensor,
                    grid_dim=num_blocks,
                    block_dim=gpu_block_size,
                )
                ctx.synchronize()
                
                # Copy row B to host
                ctx.enqueue_copy(host_row, dev_row_b)
                ctx.synchronize()
            else:
                # Even rows: B -> A
                ctx.enqueue_function_checked[automaton_row_kernel, automaton_row_kernel](
                    row_b_tensor,
                    row_a_tensor,
                    patterns_tensor,
                    grid_dim=num_blocks,
                    block_dim=gpu_block_size,
                )
                ctx.synchronize()
                
                # Copy row A to host
                ctx.enqueue_copy(host_row, dev_row_a)
                ctx.synchronize()
            
            # Transfer from host buffer to CPU grid
            var row_offset = row * self.width
            for col in range(WIDTH):
                self.cells[row_offset + col] = UInt8(host_row[col])
        
        var compute_end = py_time.time()
        var compute_duration = Float64(py_operator.sub(compute_end, compute_start))
        var transfer_duration = compute_duration  # In ping-pong, compute and transfer are interleaved
        
        var total_end = py_time.time()
        var total_duration = Float64(py_operator.sub(total_end, prep_start))
        
        self.logger.log("Ping-pong processing complete")
        return GPUTimingResult(prep_duration, compute_duration, transfer_duration, total_duration, 1)

    @staticmethod
    fn _get_num_blocks() -> Int:
        """Calculate number of GPU blocks needed to cover WIDTH columns."""
        return ceildiv(WIDTH, gpu_block_size)

    @staticmethod
    fn _init_patterns_buffer(mut host_patterns: HostBuffer[cell_dtype], allowed: List[Int]):
        """Initialize patterns host buffer with allowed patterns, padding unused slots with -1."""
        for i in range(max_patterns):
            if i < len(allowed):
                host_patterns[i] = Int32(allowed[i])
            else:
                host_patterns[i] = Int32(-1)

    @staticmethod
    fn benchmark_native_gpu(rule: Rule, logger: Logger) raises -> GPUTimingResult:
        """Benchmark native GPU without allocating CPU grid.
        
        This is a GPU-only benchmark that allocates directly on GPU without
        creating a CPU-side grid. Use this for pure GPU performance testing
        or when CPU memory is limited.
        """
        var py_time = Python.import_module("time")
        var py_builtins = Python.import_module("builtins")
        var py_operator = Python.import_module("operator")
        
        @parameter
        if not has_accelerator():
            logger.log("No GPU accelerator detected")
            print("No GPU accelerator available for benchmark")
            return GPUTimingResult(0.0, 0.0, 0.0, 0.0, 0)
        
        logger.log("Starting GPU-only benchmark (no CPU grid allocation)")
        var prep_start = py_time.time()
        
        # Build allowed patterns (only needs rule, no grid)
        var allowed = Grid._build_allowed_patterns_static(rule)
        logger.log_int("Built allowed patterns, count:", len(allowed))
        
        # Create device context
        logger.log("Creating GPU device context")
        var ctx = DeviceContext()
        
        # Allocate GPU buffers ONLY - no CPU grid!
        logger.log_int("Allocating host patterns buffer, size:", max_patterns)
        var host_patterns = ctx.enqueue_create_host_buffer[cell_dtype](max_patterns)
        logger.log_int("Allocating device grid buffer, size:", grid_size)
        var dev_grid = ctx.enqueue_create_buffer[cell_dtype](grid_size)
        logger.log_int("Allocating device patterns buffer, size:", max_patterns)
        var dev_patterns = ctx.enqueue_create_buffer[cell_dtype](max_patterns)
        logger.log("Synchronizing allocations")
        ctx.synchronize()
        logger.log("All GPU allocations successful")
        
        # Initialize patterns on host (pad with -1 for unused slots)
        Grid._init_patterns_buffer(host_patterns, allowed)
        
        # Copy patterns to device
        ctx.enqueue_copy(dev_patterns, host_patterns)
        ctx.synchronize()
        
        # Create tensor view for the grid
        var grid_tensor = LayoutTensor[cell_dtype, grid_layout](dev_grid)
        
        var prep_end = py_time.time()
        var prep_duration = Float64(py_operator.sub(prep_end, prep_start))
        
        var compute_start = py_time.time()
        
        # Calculate grid dimensions for kernel launch
        var num_blocks = Grid._get_num_blocks()
        var patterns_tensor = LayoutTensor[cell_dtype, patterns_layout](dev_patterns)
        
        # First, set the initial cell (run init kernel with 1 thread)
        ctx.enqueue_function_checked[init_center_kernel, init_center_kernel](
            grid_tensor,
            grid_dim=1,
            block_dim=1,
        )
        
        # Process ALL rows without syncing - just enqueue all kernels
        for row in range(1, HEIGHT):
            ctx.enqueue_function_checked[automaton_grid_kernel, automaton_grid_kernel](
                grid_tensor,
                patterns_tensor,
                row,
                grid_dim=num_blocks,
                block_dim=gpu_block_size,
            )
        
        # Single sync after all kernels are enqueued
        ctx.synchronize()
        
        var compute_end = py_time.time()
        var compute_duration = Float64(py_operator.sub(compute_end, compute_start))
        
        var total_end = py_time.time()
        var total_duration = Float64(py_operator.sub(total_end, prep_start))
        var transfer_duration: Float64 = 0.0  # No transfer in GPU-only mode
        
        return GPUTimingResult(prep_duration, compute_duration, transfer_duration, total_duration, 1)

    @staticmethod
    fn _build_allowed_patterns_static(rule: Rule) -> List[Int]:
        """Build allowed patterns from rule (static version for GPU-only benchmark)."""
        var allowed = List[Int]()
        for group in range(len(rule.pattern_groups)):
            for idx in range(len(rule.pattern_groups[group])):
                allowed.append(Grid._pattern_to_int_static(rule.pattern_groups[group][idx]))
        return allowed^

    @staticmethod
    fn _pattern_to_int_static(pattern: String) -> Int:
        """Convert pattern string to integer (static version)."""
        var value: Int = 0
        for idx in range(len(pattern)):
            value = value << 1
            if pattern[idx] == "1":
                value += 1
        return value

    fn _apply_rule_cpu_row(mut self, row: Int, rule: Rule):
        var prev_offset = (row - 1) * self.width
        var curr_offset = row * self.width
        for col in range(1, self.width - 1):
            var left = Int(self.cells[prev_offset + col - 1])
            var center = Int(self.cells[prev_offset + col])
            var right = Int(self.cells[prev_offset + col + 1])
            self.cells[curr_offset + col] = UInt8(rule.apply(left, center, right))

    fn _apply_rule_cpu_row_bounded(mut self, row: Int, rule: Rule, left_bound: Int, right_bound: Int):
        var prev_offset = (row - 1) * self.width
        var curr_offset = row * self.width
        for col in range(left_bound, right_bound + 1):
            if col >= 1 and col < self.width - 1:
                var left = Int(self.cells[prev_offset + col - 1])
                var center = Int(self.cells[prev_offset + col])
                var right = Int(self.cells[prev_offset + col + 1])
                self.cells[curr_offset + col] = UInt8(rule.apply(left, center, right))

    fn _init_gpu_grid(self, cp: PythonObject, builtins: PythonObject, operator: PythonObject) raises -> PythonObject:
        var grid_shape = builtins.tuple([self.height, self.width])
        var grid_gpu = cp.zeros(grid_shape, cp.int32)
        var center_index = builtins.tuple([0, self.width // 2])
        operator.setitem(grid_gpu, center_index, 1)
        return grid_gpu
    
    fn _create_allowed_patterns_array(self, rule: Rule, cp: PythonObject, builtins: PythonObject) raises -> PythonObject:
        var allowed_patterns = Self._build_allowed_patterns_static(rule)
        var py_allowed = builtins.list()
        for idx in range(len(allowed_patterns)):
            py_allowed.append(allowed_patterns[idx])
        return cp.asarray(py_allowed, cp.int32)
    
    fn _compute_rows_on_gpu(self, grid_gpu: PythonObject, allowed_array: PythonObject, cp: PythonObject, operator: PythonObject) raises:
        for row in range(1, self.height):
            var prev_row = grid_gpu.__getitem__(row - 1)
            var left = cp.roll(prev_row, 1)
            var center = prev_row
            var right = cp.roll(prev_row, -1)
            
            var codes = left * 4 + center * 2 + right
            var next_row = cp.isin(codes, allowed_array).astype(cp.int32)
            
            operator.setitem(next_row, 0, 0)
            operator.setitem(next_row, self.width - 1, 0)
            
            operator.setitem(grid_gpu, row, next_row)
    
    fn _copy_gpu_results_to_cpu(mut self, grid_gpu: PythonObject) raises:
        var result_gpu = grid_gpu.tolist()
        for row in range(self.height):
            var row_offset = row * self.width
            var row_values = result_gpu.__getitem__(row)
            for col in range(self.width):
                var value_py = row_values.__getitem__(col)
                if value_py == 0:
                    self.cells[row_offset + col] = 0
                else:
                    self.cells[row_offset + col] = 1
