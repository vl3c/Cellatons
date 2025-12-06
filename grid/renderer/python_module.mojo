"""Renderer that delegates to Python viewer module.

Mojo passes grid pointer to Python, all numpy/pygame rendering in pure Python.
This approach minimizes Python interop overhead by keeping render logic in Python.
"""

from shared.renderer.base import RendererConfig
from shared.renderer.python_module import PythonModuleBridge
from grid.grid import DISPLAY_WIDTH, DISPLAY_HEIGHT


struct PythonModuleRenderer:
    """Renders by delegating to Python viewer.py module."""
    var config: RendererConfig
    var bridge: PythonModuleBridge
    
    fn __init__(out self, var config: RendererConfig) raises:
        self.config = config^
        self.bridge = PythonModuleBridge("grid/renderer", DISPLAY_WIDTH, DISPLAY_HEIGHT)
    
    fn name(self) -> String:
        return "python_module"
    
    fn render_frame(
        mut self, 
        grid_ptr: Int, 
        grid_stride: Int,
        generation: Int = 0,
        fps: Int = 0,
        mode: String = "GPU",
        gen_time_ms: Float64 = 0.0,
    ) raises:
        """Render the full grid with status overlay."""
        var grid_np = self.bridge.ensure_grid_view(grid_ptr, grid_stride)
        
        self.bridge.get_viewer().render_frame(
            self.config.pygame,
            self.config.screen,
            grid_np,
            self.config.display_width,
            self.config.display_height,
            generation,
            fps,
            mode,
            gen_time_ms
        )

