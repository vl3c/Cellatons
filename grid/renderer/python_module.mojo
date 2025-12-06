"""Renderer that delegates to Python viewer module.

Mojo passes grid pointer to Python, all numpy/pygame rendering in pure Python.
This approach minimizes Python interop overhead by keeping render logic in Python.
"""

from python import Python, PythonObject
from grid.grid import SCREEN_WIDTH, SCREEN_HEIGHT
from grid.renderer.base import RendererConfig


struct PythonModuleRenderer:
    """Renders by delegating to Python viewer.py module.
    
    The grid data stays in Mojo memory - Python receives a pointer
    and creates a zero-copy numpy view over it.
    """
    var config: RendererConfig
    var viewer: PythonObject
    var grid_np: PythonObject
    var initialized: Bool
    var last_ptr: Int  # Track last pointer to detect buffer swap
    
    fn __init__(out self, var config: RendererConfig) raises:
        self.config = config^
        self.viewer = Self._import_viewer()
        self.grid_np = Python.none()
        self.initialized = False
        self.last_ptr = 0
    
    @staticmethod
    fn _import_viewer() raises -> PythonObject:
        """Import the Python viewer module."""
        var sys = Python.import_module("sys")
        sys.path.append("grid/renderer")
        return Python.import_module("viewer")
    
    fn name(self) -> String:
        return "python_module"
    
    fn _ensure_grid_view(
        mut self,
        grid_ptr: Int,
        grid_stride: Int,
    ) raises:
        """Initialize or update numpy view over grid buffer."""
        # Re-create view if pointer changed (buffer swap)
        if not self.initialized or self.last_ptr != grid_ptr:
            self.grid_np = self.viewer.create_grid_view(
                grid_ptr,
                grid_stride,
                SCREEN_HEIGHT,
                SCREEN_WIDTH
            )
            self.last_ptr = grid_ptr
            self.initialized = True
    
    fn render_frame(
        mut self, 
        grid_ptr: Int, 
        grid_stride: Int,
        generation: Int = 0,
        fps: Int = 0,
        mode: String = "GPU",
        gen_time_ms: Float64 = 0.0,
    ) raises:
        """Render the full grid with status overlay.
        
        Args:
            grid_ptr: Integer pointer value to grid cell data
            grid_stride: Row stride (aligned width)
            generation: Current generation number
            fps: Current frames per second
            mode: Compute mode ("GPU" or "CPU")
            gen_time_ms: Last generation time in milliseconds
        """
        self._ensure_grid_view(grid_ptr, grid_stride)
        
        self.viewer.render_frame(
            self.config.pygame,
            self.config.screen,
            self.grid_np,
            self.config.display_width,
            self.config.display_height,
            generation,
            fps,
            mode,
            gen_time_ms
        )

