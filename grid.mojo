from python import Python, PythonObject
from rule import Rule
from algorithm import parallelize
from sys.info import has_nvidia_gpu_accelerator
from sys import has_accelerator
from common import CELL_SIZE, WIDTH, HEIGHT
from gpu_timing_result import GPUTimingResult
from math import ceildiv

# Native GPU imports
from gpu.host import DeviceContext
from layout import LayoutTensor
from gpu_kernels import (
    cell_dtype,
    gpu_block_size,
    grid_size,
    grid_layout,
    max_patterns,
    patterns_layout,
    init_center_kernel,
    automaton_grid_kernel,
)


struct Grid(Copyable, Movable):
    var cells: List[List[Int]]
    var width: Int
    var height: Int
    
    fn __init__(out self, width: Int, height: Int):
        self.width = width
        self.height = height
        self.cells = List[List[Int]]()
        
        for _ in range(height):
            var row = List[Int]()
            for _ in range(width):
                row.append(0)
            self.cells.append(row^)
    
    fn __copyinit__(out self, existing: Self):
        self.width = existing.width
        self.height = existing.height
        self.cells = List[List[Int]]()
        for i in range(len(existing.cells)):
            var row = List[Int]()
            for j in range(len(existing.cells[i])):
                row.append(existing.cells[i][j])
            self.cells.append(row^)
    
    fn __moveinit__(out self, deinit existing: Self):
        self.width = existing.width
        self.height = existing.height
        self.cells = existing.cells^
    
    fn set_cell(mut self, row: Int, col: Int, value: Int):
        self.cells[row][col] = value
    
    fn get_cell(self, row: Int, col: Int) -> Int:
        return self.cells[row][col]
    
    fn get_width(self) -> Int:
        return self.width
    
    fn get_height(self) -> Int:
        return self.height
    
    fn generate_parallel_cpu(mut self, rule: Rule):
        var center = self.width // 2
        self.cells[0][center] = 1
        
        var left_bound = center
        var right_bound = center
        
        for row in range(1, self.height):
            # Expand bounds by 1 on each side, clamped to grid edges
            left_bound = left_bound - 1 if left_bound > 1 else 1
            right_bound = right_bound + 1 if right_bound < self.width - 2 else self.width - 2
            var bound_width = right_bound - left_bound + 1
            var lb = left_bound  # Capture for closure
            
            @parameter
            fn compute_cell(offset: Int):
                var col = lb + offset
                var left = self.cells[row - 1][col - 1]
                var center_val = self.cells[row - 1][col]
                var right = self.cells[row - 1][col + 1]
                self.cells[row][col] = rule.apply(left, center_val, right)
            
            parallelize[compute_cell](bound_width)
    
    fn generate_sequential_cpu(mut self, rule: Rule):
        var center = self.width // 2
        self.cells[0][center] = 1
        
        var left_bound = center
        var right_bound = center
        
        for row in range(1, self.height):
            # Expand bounds by 1 on each side, clamped to grid edges
            left_bound = left_bound - 1 if left_bound > 1 else 1
            right_bound = right_bound + 1 if right_bound < self.width - 2 else self.width - 2
            self._apply_rule_cpu_row_bounded(row, rule, left_bound, right_bound)

    fn generate_parallel_cells_cupy_gpu(mut self, rule: Rule) raises -> GPUTimingResult:
        # Set initial cell in the middle of the first row
        self.cells[0][self.width // 2] = 1
        
        var py_time = Python.import_module("time")
        var py_builder = Python.import_module("builtins")
        var py_operator = Python.import_module("operator")
        var zero = py_builder.float(0.0)
        var prep_duration = zero
        var compute_duration = zero
        var transfer_duration = zero
        var total_duration = zero
        var runs = 0
        # Check if NVIDIA GPU is available
        if has_nvidia_gpu_accelerator():
            var cp = Python.import_module("cupy")
            
            var prep_start = py_time.time()
            var grid_gpu = self._init_gpu_grid(cp, py_builder, py_operator)
            var allowed_array = self._create_allowed_patterns_array(rule, cp, py_builder)
            var prep_end = py_time.time()
            prep_duration = py_operator.sub(prep_end, prep_start)
            
            var compute_start = py_time.time()
            self._compute_rows_on_gpu(grid_gpu, allowed_array, cp, py_operator)
            var compute_end = py_time.time()
            compute_duration = py_operator.sub(compute_end, compute_start)
            
            var total_end = py_time.time()
            total_duration = py_operator.sub(total_end, prep_start)
            runs = 1
        else:
            print("No NVIDIA GPU detected, using CPU fallback")
            # CPU fallback
            for row in range(1, self.height):
                self._apply_rule_cpu_row(row, rule)
        
        return GPUTimingResult(prep_duration, compute_duration, transfer_duration, total_duration, runs)

    fn generate_native_gpu(mut self, rule: Rule) raises -> GPUTimingResult:
        """Generate cellular automaton using native Mojo GPU."""
        var py_time = Python.import_module("time")
        var py_builtins = Python.import_module("builtins")
        var py_operator = Python.import_module("operator")
        var zero = py_builtins.float(0.0)
        var prep_duration = zero
        var compute_duration = zero
        var transfer_duration = zero
        var total_duration = zero
        var runs = 0
        
        @parameter
        if not has_accelerator():
            print("No GPU accelerator, using CPU fallback")
            self.generate_sequential_cpu(rule)
        else:
            var prep_start = py_time.time()
            
            # Build allowed patterns
            var allowed = self._build_allowed_patterns(rule)
            var num_patterns = len(allowed)
            
            # Create device context
            var ctx = DeviceContext()
            
            # Allocate all buffers (host + device)
            var host_patterns = ctx.enqueue_create_host_buffer[cell_dtype](max_patterns)
            var dev_grid = ctx.enqueue_create_buffer[cell_dtype](grid_size)
            var dev_patterns = ctx.enqueue_create_buffer[cell_dtype](max_patterns)
            ctx.synchronize()  # Wait for all allocations
            
            # Initialize patterns on host (pad with -1 for unused slots)
            for i in range(max_patterns):
                if i < num_patterns:
                    host_patterns[i] = Int32(allowed[i])
                else:
                    host_patterns[i] = Int32(-1)
            
            # Copy patterns to device
            ctx.enqueue_copy(dev_patterns, host_patterns)
            ctx.synchronize()  # Wait for copy before kernel launch
            
            # Create tensor view for the grid
            var grid_tensor = LayoutTensor[cell_dtype, grid_layout](dev_grid)
            
            var prep_end = py_time.time()
            prep_duration = py_operator.sub(prep_end, prep_start)
            
            var compute_start = py_time.time()
            
            # Calculate grid dimensions for kernel launch
            var num_blocks = ceildiv(WIDTH, gpu_block_size)
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
                    num_patterns,
                    row,
                    grid_dim=num_blocks,
                    block_dim=gpu_block_size,
                )
            
            # Single sync after all kernels are enqueued
            ctx.synchronize()
            
            var compute_end = py_time.time()
            compute_duration = py_operator.sub(compute_end, compute_start)
            
            # Note: We skip GPU→CPU transfer since we use CPU results for rendering
            # The grid stays on GPU and is discarded (benchmark only)
            
            var total_end = py_time.time()
            total_duration = py_operator.sub(total_end, prep_start)
            runs = 1
        
        return GPUTimingResult(prep_duration, compute_duration, transfer_duration, total_duration, runs)

    fn _apply_rule_cpu_row(mut self, row: Int, rule: Rule):
        for col in range(1, self.width - 1):
            var left = self.cells[row - 1][col - 1]
            var center = self.cells[row - 1][col]
            var right = self.cells[row - 1][col + 1]
            
            self.cells[row][col] = rule.apply(left, center, right)

    fn _apply_rule_cpu_row_bounded(mut self, row: Int, rule: Rule, left_bound: Int, right_bound: Int):
        for col in range(left_bound, right_bound + 1):
            if col >= 1 and col < self.width - 1:
                var left = self.cells[row - 1][col - 1]
                var center = self.cells[row - 1][col]
                var right = self.cells[row - 1][col + 1]
                self.cells[row][col] = rule.apply(left, center, right)

    fn _init_gpu_grid(self, cp: PythonObject, builtins: PythonObject, operator: PythonObject) raises -> PythonObject:
        var grid_shape = builtins.tuple([self.height, self.width])
        var grid_gpu = cp.zeros(grid_shape, cp.int32)
        var center_index = builtins.tuple([0, self.width // 2])
        operator.setitem(grid_gpu, center_index, 1)
        return grid_gpu
    
    fn _create_allowed_patterns_array(self, rule: Rule, cp: PythonObject, builtins: PythonObject) raises -> PythonObject:
        var allowed_patterns = self._build_allowed_patterns(rule)
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
            var row_values = result_gpu.__getitem__(row)
            for col in range(self.width):
                var value_py = row_values.__getitem__(col)
                if value_py == 0:
                    self.cells[row][col] = 0
                else:
                    self.cells[row][col] = 1
    
    fn _pattern_to_int(self, pattern: String) -> Int:
        var value: Int = 0
        for idx in range(len(pattern)):
            value = value << 1
            if pattern[idx] == "1":
                value += 1
        return value
    
    fn _build_allowed_patterns(self, rule: Rule) -> List[Int]:
        var allowed = List[Int]()
        for group in range(len(rule.pattern_groups)):
            for idx in range(len(rule.pattern_groups[group])):
                allowed.append(self._pattern_to_int(rule.pattern_groups[group][idx]))
        return allowed^
