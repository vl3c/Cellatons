"""Renderer bridge that delegates to Python viewer module for hypercube."""

from python import Python, PythonObject
from shared.renderer.base import RendererConfig


struct HypercubePythonBridge:
    """Manages Python viewer import and zero-copy numpy view creation."""
    var viewer: PythonObject
    var grid_np: PythonObject
    var initialized: Bool
    var last_ptr: Int
    var width: Int
    var height: Int
    var depth: Int
    var w_dim: Int
    
    fn __init__(
        out self,
        module_dir: String,
        width: Int,
        height: Int,
        depth: Int,
        w_dim: Int,
        module_name: String = "viewer",
    ) raises:
        self.viewer = Self._import_viewer(module_dir, module_name)
        self.grid_np = Python.none()
        self.initialized = False
        self.last_ptr = 0
        self.width = width
        self.height = height
        self.depth = depth
        self.w_dim = w_dim
    
    @staticmethod
    fn _import_viewer(module_dir: String, module_name: String) raises -> PythonObject:
        var sys = Python.import_module("sys")
        sys.path.append(module_dir)
        return Python.import_module(module_name)
    
    fn ensure_grid_view(
        mut self,
        grid_ptr: Int,
        grid_stride: Int,
    ) raises -> PythonObject:
        """Ensure numpy view exists and is refreshed on buffer swap."""
        if not self.initialized or self.last_ptr != grid_ptr:
            self.grid_np = self.viewer.create_grid_view(
                grid_ptr,
                grid_stride,
                self.height,
                self.width,
                self.depth,
                self.w_dim,
            )
            self.last_ptr = grid_ptr
            self.initialized = True
        
        return self.grid_np
    
    fn get_viewer(self) -> PythonObject:
        return self.viewer


struct PythonModuleRenderer:
    var config: RendererConfig
    var bridge: HypercubePythonBridge
    var width: Int
    var height: Int
    var depth: Int
    var w_dim: Int
    
    fn __init__(
        out self,
        var config: RendererConfig,
        width: Int,
        height: Int,
        depth: Int,
        w_dim: Int,
    ) raises:
        self.config = config^
        self.width = width
        self.height = height
        self.depth = depth
        self.w_dim = w_dim
        self.bridge = HypercubePythonBridge("hypercube/renderer", width, height, depth, w_dim)
    
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
        render_mode: Int = 0,
        slice_index: Int = 0,
        width_logical: Int = 0,
        height_logical: Int = 0,
        depth_logical: Int = 0,
        w_dim: Int = 0,
        rotation_enabled: Bool = True,
    ) raises:
        """Render the hypercube with status overlay and mode controls."""
        var grid_np = self.bridge.ensure_grid_view(grid_ptr, grid_stride)
        
        var logical_w = w_dim if w_dim > 0 else self.w_dim
        var logical_d = depth_logical if depth_logical > 0 else self.depth
        var logical_h = height_logical if height_logical > 0 else self.height
        var logical_x = width_logical if width_logical > 0 else self.width
        
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
            render_mode,
            slice_index,
            logical_x,
            logical_h,
            logical_d,
            logical_w,
            rotation_enabled,
        )


