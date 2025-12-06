"""Grid Game of Life - Live Viewer

A 2D cellular automaton where all cells update simultaneously based on
neighbor counts. Renders fullscreen at 1440p, one generation per frame.

Rules:
- Any live cell with 2-3 live neighbors survives
- Any dead cell with exactly 3 live neighbors becomes alive
- All other cells die or stay dead

Controls:
- SPACE: Pause/Resume
- R: Reset (randomize grid)
- G: Toggle GPU/CPU mode
- Q/ESC: Quit

Run: pixi run mojo grid/main.mojo
"""

from python import Python, PythonObject
from sys import has_accelerator
from grid.grid import Grid, DISPLAY_WIDTH, DISPLAY_HEIGHT, INITIAL_DENSITY
from grid.cpu_compute import CPUCompute
from shared.renderer.base import RendererConfig, init_display
from grid.renderer import PythonModuleRenderer
from shared.benchmarking import BenchmarkStats, BenchmarkResult, print_benchmark_summary

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

alias MODE_GPU: Int = 0
alias MODE_CPU: Int = 1
alias TARGET_FPS: Int = 60


# ─────────────────────────────────────────────────────────────────────────────
# Viewer State
# ─────────────────────────────────────────────────────────────────────────────


struct ViewerState:
    """Encapsulates all mutable viewer state."""
    var running: Bool
    var paused: Bool
    var mode: Int
    var reset_requested: Bool
    var mode_changed: Bool
    var gpu_needs_upload: Bool
    var generation: Int
    var last_gen_time_ms: Float64
    
    fn __init__(out self, initial_mode: Int):
        self.running = True
        self.paused = False
        self.mode = initial_mode
        self.reset_requested = False
        self.mode_changed = False
        self.gpu_needs_upload = True
        self.generation = 0
        self.last_gen_time_ms = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────


fn _print_banner() raises:
    """Print startup banner with configuration info."""
    print("=" * 60)
    print("Grid Game of Life")
    print("=" * 60)
    print("Grid:", DISPLAY_WIDTH, "x", DISPLAY_HEIGHT, "(", DISPLAY_WIDTH * DISPLAY_HEIGHT, "cells)")
    print("Initial density:", Int(INITIAL_DENSITY * 100), "%")
    print("Target FPS:", TARGET_FPS)
    print()
    print("Controls:")
    print("  SPACE - Pause/Resume")
    print("  R     - Reset (new random grid)")
    print("  G     - Toggle GPU/CPU mode")
    print("  Q/ESC - Quit")
    print()


fn _handle_events(
    pygame: PythonObject,
    mut state: ViewerState,
    has_gpu: Bool,
) raises:
    """Process pygame events for keyboard input."""
    for event in pygame.event.get():
        var event_type = Int(event.type)
        
        if event_type == Int(pygame.QUIT):
            state.running = False
        elif event_type == Int(pygame.KEYDOWN):
            var key = Int(event.key)
            
            if key == Int(pygame.K_q) or key == Int(pygame.K_ESCAPE):
                state.running = False
            elif key == Int(pygame.K_SPACE):
                state.paused = not state.paused
                if state.paused:
                    print("Paused")
                else:
                    print("Resumed")
            elif key == Int(pygame.K_r):
                state.reset_requested = True
                print("Resetting grid...")
            elif key == Int(pygame.K_g):
                if has_gpu:
                    state.mode = 1 - state.mode
                    state.mode_changed = True
                    if state.mode == MODE_GPU:
                        print("Switched to GPU mode")
                    else:
                        print("Switched to CPU mode")
                else:
                    print("No GPU available, staying in CPU mode")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────


fn run_viewer() raises:
    """Run the Grid Game of Life viewer."""
    from grid.gpu_compute import GPUCompute
    
    _print_banner()
    
    var py_time = Python.import_module("time")
    
    # Initialize grid
    print("Initializing grid...")
    var grid = Grid(DISPLAY_WIDTH, DISPLAY_HEIGHT)
    print("Randomizing with density", Int(INITIAL_DENSITY * 100), "%...")
    grid.randomize(INITIAL_DENSITY)
    
    # Check GPU availability
    var has_gpu = grid.has_gpu()
    var initial_mode = MODE_GPU if has_gpu else MODE_CPU
    
    # Initialize GPU compute
    var gpu_compute: GPUCompute
    if has_gpu:
        print("GPU detected, initializing persistent GPU buffers...")
        gpu_compute = GPUCompute(grid.width, grid.height, grid.stride)
        print("GPU mode ready")
    else:
        print("No GPU detected, using CPU SIMD mode")
        gpu_compute = GPUCompute(1, 1, 1)
    
    # Initialize display
    print("Initializing display...")
    var config = init_display(
        "Grid Automata",
        display_width=DISPLAY_WIDTH,
        display_height=DISPLAY_HEIGHT,
        fullscreen=True,
        noframe=False,
        grid_width=DISPLAY_WIDTH,
        grid_height=DISPLAY_HEIGHT,
    )
    var pygame = config.pygame
    var clock = config.clock
    print("Display:", config.display_width, "x", config.display_height)
    print()
    print("Starting simulation...")
    print()
    
    # Create renderer
    var renderer = PythonModuleRenderer(config^)
    
    # Initialize state
    var state = ViewerState(initial_mode)
    
    # Initialize benchmark stats
    var gpu_stats = BenchmarkStats("GPU")
    var cpu_stats = BenchmarkStats("CPU")
    var frame_stats = BenchmarkStats("Frame")
    
    # Main loop
    while state.running:
        var frame_start = py_time.time()
        
        # Handle events
        _handle_events(pygame, state, has_gpu)
        
        # Handle reset
        if state.reset_requested:
            grid.randomize(INITIAL_DENSITY)
            state.generation = 0
            state.gpu_needs_upload = True
            state.reset_requested = False
            print("Grid reset complete")
        
        # Handle mode change
        if state.mode_changed:
            if state.mode == MODE_GPU:
                state.gpu_needs_upload = True
            state.mode_changed = False
        
        # Compute next generation
        if not state.paused:
            var gen_start = py_time.time()
            
            if state.mode == MODE_GPU and has_gpu:
                # Upload CPU state to GPU if needed
                if state.gpu_needs_upload:
                    var src_ptr = grid.cells_a.unsafe_ptr() if grid.active == 0 else grid.cells_b.unsafe_ptr()
                    gpu_compute.upload_from_cpu(src_ptr, grid.active)
                    state.gpu_needs_upload = False
                
                # Compute on GPU
                gpu_compute.step()
                
                # Download result to CPU for rendering
                if gpu_compute.gpu_active == 0:
                    gpu_compute.download_to_cpu(grid.cells_a)
                else:
                    gpu_compute.download_to_cpu(grid.cells_b)
                grid.active = gpu_compute.gpu_active
                
                var gen_time = Float64(py_time.time() - gen_start) * 1000.0
                state.last_gen_time_ms = gen_time
                gpu_stats.add(gen_time)
            else:
                CPUCompute.step(grid)
                state.gpu_needs_upload = True
                var gen_time = Float64(py_time.time() - gen_start) * 1000.0
                state.last_gen_time_ms = gen_time
                cpu_stats.add(gen_time)
            
            state.generation += 1
        
        # Render frame
        var fps_val = Int(clock.get_fps())
        var mode_str = "GPU" if state.mode == MODE_GPU else "CPU"
        renderer.render_frame(
            grid.get_active_cells_ptr(),
            grid.stride,
            state.generation,
            fps_val,
            mode_str,
            state.last_gen_time_ms
        )
        
        pygame.display.flip()
        _ = clock.tick(TARGET_FPS)
        
        # Track frame time
        var frame_time = Float64(py_time.time() - frame_start) * 1000.0
        frame_stats.add(frame_time)
    
    # Cleanup
    pygame.quit()
    
    # Print session stats
    var result = BenchmarkResult(
        gpu_stats^, cpu_stats^, frame_stats^,
        DISPLAY_WIDTH, DISPLAY_HEIGHT
    )
    print_benchmark_summary(result)
    print("Viewer closed.")
    print()
    print("For comprehensive benchmarks, run: pixi run mojo grid/run_benchmark.mojo")


fn main() raises:
    run_viewer()
