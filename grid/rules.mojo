"""Grid Game of Life rules.

The fundamental rules that determine cell state transitions:
- Any live cell with 2-3 live neighbors survives
- Any dead cell with exactly 3 live neighbors becomes alive
- All other cells die or stay dead

These rules are used by both CPU and GPU computation paths.
Note: GPU kernels inline these rules for performance.
"""


@always_inline
fn apply_grid_rules(current: Int, neighbors: Int) -> UInt8:
    """Apply Grid Game of Life rules.
    
    Args:
        current: Current cell state (0=dead, 1=alive)
        neighbors: Count of live neighbors (0-8)
    
    Returns:
        1 if cell should be alive next generation, 0 otherwise.
    """
    if current == 1 and (neighbors == 2 or neighbors == 3):
        return 1
    elif current == 0 and neighbors == 3:
        return 1
    return 0


@always_inline
fn should_be_alive(current: Bool, neighbors: Int) -> Bool:
    """Alternative rules interface using Bool.
    
    Args:
        current: True if cell is currently alive
        neighbors: Count of live neighbors (0-8)
    
    Returns:
        True if cell should be alive next generation.
    """
    if current and (neighbors == 2 or neighbors == 3):
        return True
    elif not current and neighbors == 3:
        return True
    return False

