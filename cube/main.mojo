"""cube (3D) cellular automaton viewer (GPU-only)."""

from python import Python, PythonObject
from cube.grid import Grid, cube_WIDTH, cube_HEIGHT, cube_DEPTH, INITIAL_DENSITY
from cube.renderer import RendererConfig, init_display, PythonModuleRenderer

alias TARGET_FPS: Int = 60


struct ViewerState:
    var running: Bool
    var paused: Bool
    var gpu_needs_upload: Bool
    var reset_requested: Bool
    var generation: Int
    var last_gen_time_ms: Float64
    
    fn __init__(out self):
        self.running = True
        self.paused = False
        self.gpu_needs_upload = True
        self.reset_requested = False
        self.generation = 0
        self.last_gen_time_ms = 0.0


fn _print_banner():
    print("=" * 60)
    print("cube Automaton (3D, GPU-only)")
    print("=" * 60)
    print("Grid:", cube_WIDTH, "x", cube_HEIGHT, "x", cube_DEPTH)
    print("Rule: B6 / S567 (26-neighbor)")
    print("Initial density:", Int(INITIAL_DENSITY * 100), "%")
    print("Target FPS:", TARGET_FPS)
    print()
    print("Controls:")
    print("  SPACE - Pause/Resume")
    print("  R     - Reset (new random grid)")
    print("  Q/ESC - Quit")
    print()


fn _handle_events(pygame: PythonObject, mut state: ViewerState) raises:
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


fn run_viewer() raises:
    from cube.gpu_compute import GPUCompute
    
    _print_banner()
    
    var py_time = Python.import_module("time")
    
    print("Initializing grid...")
    var grid = Grid()
    print("Randomizing with density", Int(INITIAL_DENSITY * 100), "%...")
    grid.randomize(INITIAL_DENSITY)
    
    var has_gpu = grid.has_gpu()
    if not has_gpu:
        print("No GPU detected. cube viewer requires GPU acceleration.")
        return
    
    print("GPU detected, initializing persistent buffers...")
    var gpu_compute = GPUCompute(grid.width, grid.height, grid.depth, grid.stride, grid.layer_stride)
    print("GPU mode ready")
    
    print("Initializing display...")
    var config = init_display("cube Automaton (GPU)")
    var pygame = config.pygame
    var clock = config.clock
    print("Display:", config.display_width, "x", config.display_height)
    print("Starting simulation...")
    print()
    
    var renderer = PythonModuleRenderer(config^)
    var state = ViewerState()
    
    while state.running:
        var frame_start = py_time.time()
        
        _handle_events(pygame, state)
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
            grid.width,
            grid.height,
            grid.depth,
        )
        
        pygame.display.flip()
        _ = clock.tick(TARGET_FPS)
        
        var _frame_time = Float64(py_time.time() - frame_start) * 1000.0
    
    pygame.quit()


fn main() raises:
    run_viewer()


