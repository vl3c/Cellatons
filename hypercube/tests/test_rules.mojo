"""Rule tests for the hypercube automaton (B6/S567)."""

from testing import assert_equal
from hypercube.grid import Grid


@always_inline
fn _apply_rule(current: Int, neighbors: Int) -> Int:
    if current == 1 and (neighbors == 5 or neighbors == 6 or neighbors == 7):
        return 1
    elif current == 0 and neighbors == 6:
        return 1
    return 0


fn test_birth_on_six_neighbors() raises:
    assert_equal(_apply_rule(0, 6), 1)


fn test_no_birth_on_five_or_seven() raises:
    assert_equal(_apply_rule(0, 5), 0)
    assert_equal(_apply_rule(0, 7), 0)


fn test_survival_on_five_six_seven() raises:
    assert_equal(_apply_rule(1, 5), 1)
    assert_equal(_apply_rule(1, 6), 1)
    assert_equal(_apply_rule(1, 7), 1)


fn test_death_otherwise() raises:
    assert_equal(_apply_rule(1, 4), 0)
    assert_equal(_apply_rule(1, 8), 0)


fn test_neighbor_count_full_shell() raises:
    var grid = Grid()
    var w = 1
    var z = 1
    var y = 1
    var x = 1
    
    # Fill all neighbors around (1,1,1,1) to ensure wrap and counting (3^4 - 1 = 80)
    for dw in range(-1, 2):
        for dz in range(-1, 2):
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dw == 0 and dz == 0 and dy == 0 and dx == 0:
                        continue
                    grid.cells_a[grid._idx(w + dw, z + dz, y + dy, x + dx)] = 1
    
    var count = grid.count_neighbors(w, z, y, x)
    assert_equal(count, 80)


