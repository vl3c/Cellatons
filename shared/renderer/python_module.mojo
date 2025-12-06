"""Shared Python viewer bridge for Mojo renderers."""

from python import Python, PythonObject


struct PythonModuleBridge:
    """Manages Python viewer import and zero-copy numpy view creation."""
    var viewer: PythonObject
    var grid_np: PythonObject
    var initialized: Bool
    var last_ptr: Int
    var width: Int
    var height: Int
    var depth: Int
    
    fn __init__(
        out self,
        module_dir: String,
        width: Int,
        height: Int,
        depth: Int = 0,
        module_name: String = "viewer",
    ) raises:
        self.viewer = Self._import_viewer(module_dir, module_name)
        self.grid_np = Python.none()
        self.initialized = False
        self.last_ptr = 0
        self.width = width
        self.height = height
        self.depth = depth
    
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
            if self.depth > 0:
                self.grid_np = self.viewer.create_grid_view(
                    grid_ptr,
                    grid_stride,
                    self.height,
                    self.width,
                    self.depth,
                )
            else:
                self.grid_np = self.viewer.create_grid_view(
                    grid_ptr,
                    grid_stride,
                    self.height,
                    self.width,
                )
            self.last_ptr = grid_ptr
            self.initialized = True
        
        return self.grid_np
    
    fn get_viewer(self) -> PythonObject:
        return self.viewer
    
    fn get_grid_np(self) -> PythonObject:
        return self.grid_np

