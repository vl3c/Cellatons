"""Renderer that delegates to Python viewer module.

Mojo passes grid pointer to Python, all numpy/pygame rendering in pure Python.
This approach minimizes Python interop overhead by keeping render logic in Python.

Expected performance: ~5-10ms per frame (~100-200 FPS).
"""

from python import Python, PythonObject
from shared.common import WIDTH, HEIGHT
from row.renderer.base import RendererConfig


struct PythonModuleRenderer:
    """Renders by delegating to Python viewer.py module.
    
    The grid data stays in Mojo memory - Python receives a pointer
    and creates a zero-copy numpy view over it.
    """
    var config: RendererConfig
    var viewer: PythonObject
    var grid_np: PythonObject
    var initialized: Bool
    
    fn __init__(out self, var config: RendererConfig) raises:
        self.config = config^
        self.viewer = Self._import_viewer()
        self.grid_np = Python.none()
        self.initialized = False
    
    @staticmethod
    fn _import_viewer() raises -> PythonObject:
        """Import the Python viewer module."""
        var sys = Python.import_module("sys")
        sys.path.append("row/renderer")
        return Python.import_module("viewer")
    
    fn name(self) -> String:
        return "python_module"
    
    fn _ensure_grid_view(
        mut self,
        grid_ptr: UnsafePointer[UInt8],
        grid_stride: Int,
    ) raises:
        """Initialize numpy view over grid buffer (lazy, called once)."""
        if self.initialized:
            return
        
        self.grid_np = self.viewer.create_grid_view(
            Int(grid_ptr),
            grid_stride,
            HEIGHT,
            WIDTH
        )
        self.initialized = True
    
    fn render_window(
        mut self, 
        grid_ptr: UnsafePointer[UInt8], 
        grid_stride: Int,
        start_row: Int,
        progress: Int = 0,
        fps: Int = 0,
    ) raises:
        """Render visible window with status overlay.
        
        Args:
            grid_ptr: Pointer to grid cell data
            grid_stride: Row stride (aligned width)
            start_row: First visible row (scroll position)
            progress: Playback progress percentage (0-100)
            fps: Current frames per second
        """
        self._ensure_grid_view(grid_ptr, grid_stride)
        
        self.viewer.render_window(
            self.config.pygame,
            self.config.screen,
            self.grid_np,
            start_row,
            self.config.display_width,
            self.config.display_height,
            self.config.view_left,
            progress,
            fps
        )
