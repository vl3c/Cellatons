"""Benchmark result container for storing complete benchmark data (shared)."""

from shared.benchmarking.stats import BenchmarkStats


struct BenchmarkResult:
    """Complete benchmark result with metadata and comparison logic.
    
    Contains GPU, CPU, and frame timing statistics along with
    grid configuration information. Supports throughput and
    speedup calculations for comparison.
    """
    var gpu_stats: BenchmarkStats
    var cpu_stats: BenchmarkStats
    var frame_stats: BenchmarkStats
    var grid_width: Int
    var grid_height: Int
    var total_cells: Int
    var num_generations: Int
    var initial_density: Float64
    
    # ─────────────────────────────────────────────────────────────────────────
    # Initialization
    # ─────────────────────────────────────────────────────────────────────────
    
    fn __init__(
        out self,
        var gpu_stats: BenchmarkStats,
        var cpu_stats: BenchmarkStats,
        var frame_stats: BenchmarkStats,
        grid_width: Int,
        grid_height: Int,
        num_generations: Int = 0,
        initial_density: Float64 = 0.0,
    ):
        """Initialize benchmark result with stats and grid dimensions."""
        self.gpu_stats = gpu_stats^
        self.cpu_stats = cpu_stats^
        self.frame_stats = frame_stats^
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.total_cells = grid_width * grid_height
        self.num_generations = num_generations
        self.initial_density = initial_density
    
    # ─────────────────────────────────────────────────────────────────────────
    # Throughput Calculations
    # ─────────────────────────────────────────────────────────────────────────
    
    fn cpu_total_time(self) -> Float64:
        """Calculate CPU total benchmark time in seconds."""
        if self.cpu_stats.count() == 0:
            return 0.0
        return self.cpu_stats.avg() * Float64(self.cpu_stats.count()) / 1000.0
    
    fn gpu_total_time(self) -> Float64:
        """Calculate GPU total benchmark time in seconds."""
        if self.gpu_stats.count() == 0:
            return 0.0
        return self.gpu_stats.avg() * Float64(self.gpu_stats.count()) / 1000.0
    
    fn cpu_generations_per_sec(self) -> Float64:
        """Calculate CPU throughput in generations per second."""
        var total = self.cpu_total_time()
        if total <= 0:
            return 0.0
        return Float64(self.cpu_stats.count()) / total
    
    fn gpu_generations_per_sec(self) -> Float64:
        """Calculate GPU throughput in generations per second."""
        var total = self.gpu_total_time()
        if total <= 0:
            return 0.0
        return Float64(self.gpu_stats.count()) / total
    
    fn cpu_cells_per_sec(self) -> Float64:
        """Calculate CPU throughput in cells per second."""
        return Float64(self.total_cells) * self.cpu_generations_per_sec()
    
    fn gpu_cells_per_sec(self) -> Float64:
        """Calculate GPU throughput in cells per second."""
        return Float64(self.total_cells) * self.gpu_generations_per_sec()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Comparison
    # ─────────────────────────────────────────────────────────────────────────
    
    fn has_gpu_results(self) -> Bool:
        """Check if GPU benchmark results are available."""
        return self.gpu_stats.count() > 0
    
    fn has_cpu_results(self) -> Bool:
        """Check if CPU benchmark results are available."""
        return self.cpu_stats.count() > 0
    
    fn has_comparison(self) -> Bool:
        """Check if both GPU and CPU results are available for comparison."""
        return self.has_gpu_results() and self.has_cpu_results()
    
    fn gpu_speedup(self) -> Float64:
        """Calculate GPU speedup over CPU.
        
        Returns value > 1 if GPU is faster, < 1 if CPU is faster.
        Returns 0 if comparison not available.
        """
        if not self.has_comparison():
            return 0.0
        if self.gpu_stats.avg() <= 0:
            return 0.0
        return self.cpu_stats.avg() / self.gpu_stats.avg()
    
    fn is_gpu_faster(self) -> Bool:
        """Check if GPU is faster than CPU."""
        return self.gpu_speedup() > 1.0
    
    fn avg_fps(self) -> Float64:
        """Calculate average FPS from frame stats."""
        if self.frame_stats.count() == 0 or self.frame_stats.avg() <= 0:
            return 0.0
        return 1000.0 / self.frame_stats.avg()

