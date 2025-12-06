"""Benchmark statistics collection for timing measurements."""


struct BenchmarkStats(Movable):
    """Stores timing statistics for benchmark analysis.
    
    Collects timing samples and provides statistical analysis methods
    including average, min, max, and standard deviation.
    """
    var times: List[Float64]
    var mode: String
    
    # ─────────────────────────────────────────────────────────────────────────
    # Initialization
    # ─────────────────────────────────────────────────────────────────────────
    
    fn __init__(out self, mode: String):
        """Initialize empty stats for given mode."""
        self.times = List[Float64]()
        self.mode = mode
    
    fn __moveinit__(out self, deinit existing: Self):
        """Move constructor."""
        self.times = existing.times^
        self.mode = existing.mode^
    
    # ─────────────────────────────────────────────────────────────────────────
    # Sample Management
    # ─────────────────────────────────────────────────────────────────────────
    
    fn add(mut self, time_ms: Float64):
        """Add a timing sample in milliseconds."""
        self.times.append(time_ms)
    
    fn clear(mut self):
        """Clear all timing samples."""
        self.times.clear()
    
    fn count(self) -> Int:
        """Return number of samples collected."""
        return len(self.times)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Statistical Analysis
    # ─────────────────────────────────────────────────────────────────────────
    
    fn avg(self) -> Float64:
        """Calculate average time in milliseconds."""
        if len(self.times) == 0:
            return 0.0
        var total: Float64 = 0.0
        for i in range(len(self.times)):
            total += self.times[i]
        return total / Float64(len(self.times))
    
    fn min_val(self) -> Float64:
        """Return minimum time in milliseconds."""
        if len(self.times) == 0:
            return 0.0
        var m = self.times[0]
        for i in range(1, len(self.times)):
            if self.times[i] < m:
                m = self.times[i]
        return m
    
    fn max_val(self) -> Float64:
        """Return maximum time in milliseconds."""
        if len(self.times) == 0:
            return 0.0
        var m = self.times[0]
        for i in range(1, len(self.times)):
            if self.times[i] > m:
                m = self.times[i]
        return m
    
    fn std_dev(self) -> Float64:
        """Calculate standard deviation in milliseconds."""
        if len(self.times) < 2:
            return 0.0
        var avg_val = self.avg()
        var sum_sq: Float64 = 0.0
        for i in range(len(self.times)):
            var diff = self.times[i] - avg_val
            sum_sq += diff * diff
        return (sum_sq / Float64(len(self.times) - 1)) ** 0.5

