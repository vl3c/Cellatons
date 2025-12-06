"""Renderer that delegates to Python viewer module.

Mojo passes grid pointer to Python, all numpy/pygame rendering in pure Python.
This approach minimizes Python interop overhead by keeping render logic in Python.

Expected performance: ~5-10ms per frame (~100-200 FPS).
"""

from shared.common import WIDTH, HEIGHT
from shared.renderer.python_module import PythonModuleBridge
from row.renderer.base import RendererConfig


struct PythonModuleRenderer:
    """Renders by delegating to Python viewer.py module."""
    var config: RendererConfig
    var bridge: PythonModuleBridge
    
    fn __init__(out self, var config: RendererConfig) raises:
        self.config = config^
        self.bridge = PythonModuleBridge("row/renderer", WIDTH, HEIGHT)
    
    fn name(self) -> String:
        return "python_module"
    
    fn render_window(
        mut self, 
        grid_ptr: UnsafePointer[UInt8], 
        grid_stride: Int,
        start_row: Int,
        progress: Int = 0,
        fps: Int = 0,
    ) raises:
        """Render visible window with status overlay."""
        var grid_np = self.bridge.ensure_grid_view(Int(grid_ptr), grid_stride)
        
        self.bridge.get_viewer().render_window(
            self.config.pygame,
            self.config.screen,
            grid_np,
            start_row,
            self.config.display_width,
            self.config.display_height,
            self.config.view_left,
            progress,
            fps
        )
