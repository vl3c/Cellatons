"""Live viewer for row cellular automata.

Uses the fastest renderer (python_module) to display
the automaton with scrolling playback.

Controls:
- SPACE: Pause/Resume
- R: Reset to start
- Q/ESC: Quit

Run: pixi run mojo row/renderer/main.mojo
"""

from python import Python, PythonObject
from shared.common import WIDTH, HEIGHT
from shared.logger import Logger
from row.grid import Grid
from row.rule import Rule
from row.renderer.base import init_display, RendererConfig
from row.renderer.python_module import PythonModuleRenderer


# Playback configuration
alias ROWS_PER_SECOND: Float64 = 200.0
alias TARGET_FPS: Int = 60


fn create_rule_30() -> Rule:
    """Create Rule 30 with proper pattern groups.
    
    Rule 30 is famous for generating chaotic, pseudo-random patterns
    and is used in Mathematica's random number generator.
    """
    var groups = List[List[String]]()
    
    var group1 = List[String]()
    group1.append("100")
    group1.append("011")
    groups.append(group1^)
    
    var group2 = List[String]()
    group2.append("010")
    group2.append("001")
    groups.append(group2^)
    
    return Rule("Rule 30", "rule_30", groups^)


fn print_banner() raises:
    """Print startup banner with configuration info."""
    print("=" * 60)
    print("row CA Live Viewer")
    print("=" * 60)
    print("Rule: 30")
    print("Grid:", WIDTH, "x", HEIGHT)
    print("Playback:", Int(ROWS_PER_SECOND), "rows/second @", TARGET_FPS, "FPS")
    print()
    print("Controls:")
    print("  SPACE - Pause/Resume")
    print("  R     - Reset to start")
    print("  Q/ESC - Quit")
    print()


fn generate_grid() raises -> Grid:
    """Generate the cellular automaton grid using fastest CPU path."""
    var py_time = Python.import_module("time")
    
    print("Generating grid with SIMD CPU...")
    var start = py_time.time()
    
    var logger = Logger()
    var rule = create_rule_30()
    var grid = Grid(WIDTH, HEIGHT, logger)
    grid.generate_simd_cpu(rule)
    
    var elapsed_ms = Float64(py_time.time() - start) * 1000.0
    print("Grid generated in", elapsed_ms, "ms")
    print()
    
    return grid^


fn handle_events(
    pygame: PythonObject,
    mut running: Bool,
    mut paused: Bool,
    mut scroll_position: Float64,
) raises:
    """Process pygame events for keyboard input."""
    for event in pygame.event.get():
        var event_type = Int(event.type)
        
        if event_type == Int(pygame.QUIT):
            running = False
        elif event_type == Int(pygame.KEYDOWN):
            var key = Int(event.key)
            
            if key == Int(pygame.K_q) or key == Int(pygame.K_ESCAPE):
                running = False
            elif key == Int(pygame.K_SPACE):
                paused = not paused
                if paused:
                    print("Paused at row", Int(scroll_position))
                else:
                    print("Resumed")
            elif key == Int(pygame.K_r):
                scroll_position = 0.0
                paused = False
                print("Reset to start")


fn run_viewer() raises:
    """Run the live viewer main loop."""
    print_banner()
    
    var grid = generate_grid()
    var grid_ptr = grid.cells.unsafe_ptr()
    var grid_stride = grid.stride
    
    # Initialize display
    var config = init_display("row CA - Rule 30")
    var pygame = config.pygame
    var clock = config.clock
    var max_scroll = Float64(config.max_scroll_rows())
    
    print("Display:", config.display_width, "x", config.display_height)
    print("Starting playback...")
    print()
    
    # Create renderer
    var renderer = PythonModuleRenderer(config^)
    
    # Playback state
    var scroll_position: Float64 = 0.0
    var rows_per_frame = ROWS_PER_SECOND / Float64(TARGET_FPS)
    var running = True
    var paused = False
    
    # Main loop
    while running:
        handle_events(pygame, running, paused, scroll_position)
        
        # Update scroll position
        if not paused:
            scroll_position += rows_per_frame
            
            if scroll_position >= max_scroll:
                scroll_position = max_scroll
                paused = True
                print()
                print("Playback complete!")
                print("  SPACE - Resume (stays at end)")
                print("  R     - Reset to start")
                print("  Q     - Quit")
        
        # Render frame
        var start_row = Int(scroll_position)
        var progress = Int((scroll_position / max_scroll) * 100.0)
        var fps_val = Int(clock.get_fps())
        renderer.render_window(grid_ptr, grid_stride, start_row, progress, fps_val)
        
        pygame.display.flip()
        _ = clock.tick(TARGET_FPS)
    
    pygame.quit()
    print("Viewer closed.")


fn main() raises:
    run_viewer()
