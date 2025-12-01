"""Debug logging utilities for the cellular automaton project.

Uses a Logger struct to track start time for relative timestamps.
All functions are thread-safe and do not use Python.
"""

from shared.common import DEBUG
from time import perf_counter_ns


struct Logger(Copyable, Movable):
    """Logger with timestamps relative to creation time."""
    var start_ns: UInt
    
    fn __init__(out self):
        """Create logger and capture start time."""
        self.start_ns = perf_counter_ns()
    
    fn __copyinit__(out self, existing: Self):
        self.start_ns = existing.start_ns
    
    fn __moveinit__(out self, deinit existing: Self):
        self.start_ns = existing.start_ns
    
    fn _format_elapsed(self) -> String:
        """Format elapsed time as seconds.milliseconds."""
        var elapsed = perf_counter_ns() - self.start_ns
        var secs = elapsed // 1_000_000_000
        var ms = (elapsed % 1_000_000_000) // 1_000_000
        
        var secs_str = String(secs)
        while len(secs_str) < 3:
            secs_str = " " + secs_str
        
        var ms_str = String(ms)
        if ms < 10:
            ms_str = "00" + ms_str
        elif ms < 100:
            ms_str = "0" + ms_str
        
        return secs_str + "." + ms_str
    
    fn log(self, msg: String):
        """Print a debug message with relative timestamp."""
        @parameter
        if DEBUG:
            print("[" + self._format_elapsed() + "s]", msg, flush=True)
    
    fn log_int(self, msg: String, value: Int):
        """Print a debug message with an integer value."""
        @parameter
        if DEBUG:
            print("[" + self._format_elapsed() + "s]", msg, value, flush=True)
