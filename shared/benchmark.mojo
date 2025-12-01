"""Benchmark result collection and summary table printing."""

from python import Python, PythonObject
from shared.common import WIDTH, HEIGHT


struct BenchmarkResult(Copyable, Movable):
    """Single benchmark timing result."""
    var name: String
    var total_time: Float64
    var gpu_time: Float64
    var has_gpu: Bool
    
    fn __init__(out self, name: String, total_time: Float64):
        self.name = name
        self.total_time = total_time
        self.gpu_time = 0.0
        self.has_gpu = False
    
    fn __init__(out self, name: String, total_time: Float64, gpu_time: Float64):
        self.name = name
        self.total_time = total_time
        self.gpu_time = gpu_time
        self.has_gpu = True
    
    fn __copyinit__(out self, existing: Self):
        self.name = existing.name
        self.total_time = existing.total_time
        self.gpu_time = existing.gpu_time
        self.has_gpu = existing.has_gpu
    
    fn __moveinit__(out self, deinit existing: Self):
        self.name = existing.name^
        self.total_time = existing.total_time
        self.gpu_time = existing.gpu_time
        self.has_gpu = existing.has_gpu


struct BenchmarkSuite:
    """Collects benchmark results and prints summary table."""
    var results: List[BenchmarkResult]
    var num_rules: Int
    var start_time: Float64
    
    fn __init__(out self, num_rules: Int) raises:
        self.results = List[BenchmarkResult]()
        self.num_rules = num_rules
        var py_time = Python.import_module("time")
        self.start_time = Float64(py_time.time())
    
    fn add(mut self, name: String, total_time: PythonObject) raises:
        """Add a CPU-only benchmark result."""
        self.results.append(BenchmarkResult(name, Float64(total_time)))
    
    fn add_gpu(mut self, name: String, total_time: PythonObject, gpu_time: PythonObject) raises:
        """Add a GPU benchmark result."""
        self.results.append(BenchmarkResult(name, Float64(total_time), Float64(gpu_time)))
    
    fn print_summary(self) raises:
        """Print formatted benchmark summary table."""
        var py_time = Python.import_module("time")
        var total_elapsed = Float64(py_time.time()) - self.start_time
        
        var cells = WIDTH * HEIGHT
        var cells_m = cells // 1_000_000
        
        # Memory calculations
        # CPU: List[List[Int]] where Int is 64-bit = 8 bytes per cell
        # GPU: Int32 = 4 bytes per cell
        var cpu_bytes = cells * 8
        var gpu_bytes = cells * 4
        var cpu_mem = Self._format_bytes(cpu_bytes)
        var gpu_mem = Self._format_bytes(gpu_bytes)
        
        print()
        print("=" * 55)
        print("BENCHMARK SUMMARY")
        print("=" * 55)
        print("Grid:", WIDTH, "x", HEIGHT, "(", cells_m, "M cells) |", self.num_rules, "rules")
        print("Memory: RAM", cpu_mem, "| VRAM", gpu_mem)
        print("-" * 55)
        print("Method                       Total       GPU")
        print("-" * 55)
        
        for i in range(len(self.results)):
            var r = self.results[i].copy()
            self._print_row(r.name, r.total_time, r.gpu_time, r.has_gpu)
        
        print("-" * 55)
        self._print_total(total_elapsed)
        print("=" * 55)
    
    fn _print_row(self, name: String, total: Float64, gpu: Float64, has_gpu: Bool) raises:
        """Print a single result row with proper formatting."""
        # Pad name to 28 chars
        var padded = name
        while len(padded) < 28:
            padded += " "
        
        # Format times
        var total_str = self._format_time(total)
        
        if has_gpu:
            var gpu_str = self._format_time(gpu)
            print(padded, total_str, "s  ", gpu_str, "s")
        else:
            print(padded, total_str, "s       -")
    
    fn _print_total(self, total: Float64) raises:
        """Print the total time row."""
        var total_str = self._format_time(total)
        print("Total:                      ", total_str, "s")
    
    @staticmethod
    fn _format_time(t: Float64) raises -> String:
        """Format time to 6.3f format."""
        var py_builtins = Python.import_module("builtins")
        return String(py_builtins.format(t, "6.3f"))
    
    @staticmethod
    fn _format_bytes(bytes: Int) -> String:
        """Format bytes to human-readable string (MB or GB)."""
        var gb = bytes // (1024 * 1024 * 1024)
        var mb = (bytes % (1024 * 1024 * 1024)) // (1024 * 1024)
        
        if gb > 0:
            if mb >= 100:
                # Round to nearest 0.1 GB
                var total_mb = bytes // (1024 * 1024)
                var gb_tenths = (total_mb * 10) // 1024
                var gb_whole = gb_tenths // 10
                var gb_frac = gb_tenths % 10
                return String(gb_whole) + "." + String(gb_frac) + " GB"
            else:
                return String(gb) + "." + String(mb // 100) + " GB"
        else:
            return String(mb) + " MB"

