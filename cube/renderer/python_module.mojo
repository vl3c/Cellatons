"""Renderer bridge that delegates to Python viewer module."""

from shared.renderer.base import RendererConfig
from shared.renderer.python_module import PythonModuleBridge
from cube.grid import cube_WIDTH, cube_HEIGHT, cube_DEPTH


struct PythonModuleRenderer:
    var config: RendererConfig
    var bridge: PythonModuleBridge
    
    fn __init__(out self, var config: RendererConfig) raises:
        self.config = config^
        self.bridge = PythonModuleBridge("cube/renderer", cube_WIDTH, cube_HEIGHT, cube_DEPTH)
    
    fn name(self) -> String:
        return "python_module"
    
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
        var grid_np = self.bridge.ensure_grid_view(grid_ptr, grid_stride)
        
        self.bridge.get_viewer().render_frame(
            self.config.pygame,
            self.config.screen,
            grid_np,
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


