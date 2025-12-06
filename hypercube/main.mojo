"""hypercube (4D) cellular automaton viewer (GPU-only)."""

from python import Python, PythonObject
from hypercube.grid import Grid, HYPER_WIDTH, HYPER_HEIGHT, HYPER_DEPTH, HYPER_W, INITIAL_DENSITY
from hypercube.renderer import PythonModuleRenderer
from shared.renderer.base import RendererConfig, init_display
from shared.display import DISPLAY_WIDTH, DISPLAY_HEIGHT

alias TARGET_FPS: Int = 60

# Render modes (cycled with 'T')
alias MODE_SLICE: Int = 0          # Single W slice
alias MODE_MAX_INTENSITY: Int = 1  # Max over W
alias MODE_TILED: Int = 2          # Multiple W slices stacked
alias MODE_OFF: Int = 3            # Rendering disabled
alias MODE_COUNT: Int = 4


struct ViewerState:
    var running: Bool
    var paused: Bool
    var gpu_needs_upload: Bool
    var reset_requested: Bool
    var generation: Int
    var last_gen_time_ms: Float64
    var render_mode: Int
    var slice_index: Int
    
    fn __init__(out self):
        self.running = True
        self.paused = False
        self.gpu_needs_upload = True
        self.reset_requested = False
        self.generation = 0
        self.last_gen_time_ms = 0.0
        self.render_mode = MODE_SLICE
        self.slice_index = 0


fn _mode_name(mode: Int) -> String:
    if mode == MODE_SLICE:
        return "Slice"
    elif mode == MODE_MAX_INTENSITY:
        return "Max-Intensity"
    elif mode == MODE_TILED:
        return "Tiled"
    return "Off"


fn _print_banner():
    print("=" * 60)
    print("hypercube Automaton (4D, GPU-only)")
    print("=" * 60)
    print("Grid:", HYPER_W, "x", HYPER_DEPTH, "x", HYPER_HEIGHT, "x", HYPER_WIDTH)
    print("Rule: B6 / S567 (80-neighbor)")
    print("Initial density:", Int(INITIAL_DENSITY * 100), "%")
    print("Render modes: Slice, Max-Intensity, Tiled, Off (toggle: T)")
    print("Target FPS:", TARGET_FPS)
    print()
    print("Controls:")
    print("  SPACE - Pause/Resume")
    print("  R     - Reset (new random grid)")
    print("  T     - Cycle render mode")
    print("  [ / ] - Move W slice (Slice mode)")
    print("  Q/ESC - Quit")
    print()


fn _handle_events(pygame: PythonObject, mut state: ViewerState, w_dim: Int) raises:
    for event in pygame.event.get():
        var etype = Int(event.type)
        if etype == Int(pygame.QUIT):
            state.running = False
        elif etype == Int(pygame.KEYDOWN):
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
                return
            elif key == Int(pygame.K_t):
                state.render_mode = (state.render_mode + 1) % MODE_COUNT
                print("Render mode:", _mode_name(state.render_mode))
            elif key == Int(pygame.K_LEFTBRACKET):
                state.slice_index = (state.slice_index - 1 + w_dim) % w_dim
            elif key == Int(pygame.K_RIGHTBRACKET):
                state.slice_index = (state.slice_index + 1) % w_dim


fn run_viewer() raises:
    from hypercube.gpu_compute import GPUCompute
    
    _print_banner()
    
    var py_time = Python.import_module("time")
    
    print("Initializing grid...")
    var grid = Grid()
    print("Randomizing with density", Int(INITIAL_DENSITY * 100), "%...")
    grid.randomize(INITIAL_DENSITY)
    
    var has_gpu = grid.has_gpu()
    if not has_gpu:
        print("No GPU detected. hypercube viewer requires GPU acceleration.")
        return
    
    print("GPU detected, initializing persistent buffers...")
    var gpu_compute = GPUCompute(
        grid.width,
        grid.height,
        grid.depth,
        grid.w_dim,
        grid.stride,
        grid.layer_stride,
        grid.hyperlayer_stride,
    )
    print("GPU mode ready")
    
    print("Initializing display...")
    var config = init_display(
        "hypercube Automaton (GPU)",
        display_width=DISPLAY_WIDTH,
        display_height=DISPLAY_HEIGHT,
        fullscreen=True,
        noframe=True,
        grid_width=DISPLAY_WIDTH,
        grid_height=DISPLAY_HEIGHT,
    )
    var pygame = config.pygame
    var clock = config.clock
    print("Display:", config.display_width, "x", config.display_height)
    print("Starting simulation...")
    print()
    
    var renderer = PythonModuleRenderer(config^, grid.width, grid.height, grid.depth, grid.w_dim)
    var state = ViewerState()
    
    while state.running:
        var frame_start = py_time.time()
        
        _handle_events(pygame, state, grid.w_dim)
        if not state.running:
            break
        
        if state.reset_requested:
            grid.randomize(INITIAL_DENSITY)
            state.generation = 0
            state.gpu_needs_upload = True
            state.reset_requested = False
        
        if not state.paused:
            var gen_start = py_time.time()
            
            if state.gpu_needs_upload:
                var src_ptr = grid.cells_a.unsafe_ptr() if grid.active == 0 else grid.cells_b.unsafe_ptr()
                gpu_compute.upload_from_cpu(src_ptr, grid.active)
                state.gpu_needs_upload = False
            
            gpu_compute.step()
            
            if gpu_compute.gpu_active == 0:
                gpu_compute.download_to_cpu(grid.cells_a)
            else:
                gpu_compute.download_to_cpu(grid.cells_b)
            grid.active = gpu_compute.gpu_active
            
            var gen_time = Float64(py_time.time() - gen_start) * 1000.0
            state.last_gen_time_ms = gen_time
            
            state.generation += 1
        
        var fps_val = Int(clock.get_fps())
        renderer.render_frame(
            grid.get_active_cells_ptr(),
            grid.stride,
            state.generation,
            fps_val,
            state.last_gen_time_ms,
            state.paused,
            state.render_mode,
            state.slice_index,
            grid.width,
            grid.height,
            grid.depth,
            grid.w_dim,
        )
        
        pygame.display.flip()
        _ = clock.tick(TARGET_FPS)
        
        var _frame_time = Float64(py_time.time() - frame_start) * 1000.0
    
    pygame.quit()


fn main() raises:
    run_viewer()


