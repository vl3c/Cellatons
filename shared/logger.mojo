"""Debug logging utilities for the cellular automaton project."""

from shared.common import DEBUG


@always_inline
fn debug_log(msg: String):
    """Print a debug message if DEBUG mode is enabled.
    
    Uses @parameter if for zero-cost when DEBUG is False - the entire
    function body is eliminated at compile time.
    """
    @parameter
    if DEBUG:
        print("[DEBUG]", msg)


@always_inline
fn debug_log_value[T: Stringable](msg: String, value: T):
    """Print a debug message with a value if DEBUG mode is enabled."""
    @parameter
    if DEBUG:
        print("[DEBUG]", msg, str(value))


@always_inline
fn debug_log_int(msg: String, value: Int):
    """Print a debug message with an integer value if DEBUG mode is enabled."""
    @parameter
    if DEBUG:
        print("[DEBUG]", msg, value)

