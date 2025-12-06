"""Renderer bridge that delegates to Python viewer module."""

from python import Python, PythonObject
from cube.grid import cube_WIDTH, cube_HEIGHT, cube_DEPTH
from cube.renderer.base import RendererConfig


struct PythonModuleRenderer:
    var config: RendererConfig
    var viewer: PythonObject
    var grid_np: PythonObject
    var initialized: Bool
    var last_ptr: Int
    
    fn __init__(out self, var config: RendererConfig) raises:
        self.config = config^
        self.viewer = Self._import_viewer()
        self.grid_np = Python.none()
        self.initialized = False
        self.last_ptr = 0
    
    @staticmethod
    fn _import_viewer() raises -> PythonObject:
        var sys = Python.import_module("sys")
        sys.path.append("cube/renderer")
        return Python.import_module("viewer")
    
    fn name(self) -> String:
        return "python_module"
    
    fn _ensure_grid_view(
        mut self,
        grid_ptr: Int,
        grid_stride: Int,
    ) raises:
        if not self.initialized or self.last_ptr != grid_ptr:
            self.grid_np = self.viewer.create_grid_view(
                grid_ptr,
                grid_stride,
                cube_HEIGHT,
                cube_WIDTH,
                cube_DEPTH,
            )
            self.last_ptr = grid_ptr
            self.initialized = True
    
    fn render_frame(
        mut self,
        grid_ptr: Int,
        grid_stride: Int,
        generation: Int = 0,
        fps: Int = 0,
        gen_time_ms: Float64 = 0.0,
        paused: Bool = False,
        width_logical: Int = cube_WIDTH,
        height_logical: Int = cube_HEIGHT,
        depth_logical: Int = cube_DEPTH,
    ) raises:
        self._ensure_grid_view(grid_ptr, grid_stride)
        
        self.viewer.render_frame(
            self.config.pygame,
            self.config.screen,
            self.grid_np,
            self.config.display_width,
            self.config.display_height,
            generation,
            fps,
            gen_time_ms,
            paused,
            width_logical,
            height_logical,
            depth_logical,
        )


