"""Shared ping-pong GPU compute helpers for 2D and 3D grids.

Provides buffer allocation, upload/download, and active-buffer tracking.
Kernel launches are done inline by callers using exposed buffers.
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from memory import UnsafePointer


struct PingPongGPU2D(Movable):
    """Ping-pong GPU buffer manager for 2D grids."""
    var ctx: DeviceContext
    var dev_a: DeviceBuffer[DType.uint8]
    var dev_b: DeviceBuffer[DType.uint8]
    var host: HostBuffer[DType.uint8]
    var grid_size: Int
    var width: Int
    var height: Int
    var stride: Int
    var gpu_active: Int
    var needs_upload: Bool
    
    fn __init__(out self, width: Int, height: Int, stride: Int) raises:
        self.width = width
        self.height = height
        self.stride = stride
        self.grid_size = stride * height
        self.gpu_active = 0
        self.needs_upload = True
        
        self.ctx = DeviceContext()
        self.dev_a = self.ctx.enqueue_create_buffer[DType.uint8](self.grid_size)
        self.dev_b = self.ctx.enqueue_create_buffer[DType.uint8](self.grid_size)
        self.host = self.ctx.enqueue_create_host_buffer[DType.uint8](self.grid_size)
        self.ctx.synchronize()
    
    fn __moveinit__(out self, deinit existing: Self):
        self.ctx = existing.ctx^
        self.dev_a = existing.dev_a^
        self.dev_b = existing.dev_b^
        self.host = existing.host^
        self.grid_size = existing.grid_size
        self.width = existing.width
        self.height = existing.height
        self.stride = existing.stride
        self.gpu_active = existing.gpu_active
        self.needs_upload = existing.needs_upload
    
    fn upload_from_cpu(
        mut self,
        cpu_ptr: UnsafePointer[UInt8],
        cpu_active: Int,
    ) raises:
        for i in range(self.grid_size):
            self.host[i] = cpu_ptr[i]
        
        if cpu_active == 0:
            self.ctx.enqueue_copy(self.dev_a, self.host)
        else:
            self.ctx.enqueue_copy(self.dev_b, self.host)
        self.ctx.synchronize()
        
        self.gpu_active = cpu_active
        self.needs_upload = False
    
    fn swap_active(mut self):
        """Swap the active buffer after a kernel step."""
        self.gpu_active = 1 - self.gpu_active
    
    fn download_to_cpu(
        mut self,
        mut dst: List[UInt8],
    ) raises:
        if self.gpu_active == 0:
            self.ctx.enqueue_copy(self.host, self.dev_a)
        else:
            self.ctx.enqueue_copy(self.host, self.dev_b)
        self.ctx.synchronize()
        
        for i in range(self.grid_size):
            dst[i] = self.host[i]
    
    fn mark_dirty(mut self):
        self.needs_upload = True


struct PingPongGPU3D(Movable):
    """Ping-pong GPU buffer manager for 3D grids."""
    var ctx: DeviceContext
    var dev_a: DeviceBuffer[DType.uint8]
    var dev_b: DeviceBuffer[DType.uint8]
    var host: HostBuffer[DType.uint8]
    var grid_size: Int
    var width: Int
    var height: Int
    var depth: Int
    var stride: Int
    var layer_stride: Int
    var gpu_active: Int
    var needs_upload: Bool
    
    fn __init__(
        out self,
        width: Int,
        height: Int,
        depth: Int,
        stride: Int,
        layer_stride: Int,
    ) raises:
        self.width = width
        self.height = height
        self.depth = depth
        self.stride = stride
        self.layer_stride = layer_stride
        self.grid_size = layer_stride * depth
        self.gpu_active = 0
        self.needs_upload = True
        
        self.ctx = DeviceContext()
        self.dev_a = self.ctx.enqueue_create_buffer[DType.uint8](self.grid_size)
        self.dev_b = self.ctx.enqueue_create_buffer[DType.uint8](self.grid_size)
        self.host = self.ctx.enqueue_create_host_buffer[DType.uint8](self.grid_size)
        self.ctx.synchronize()
    
    fn __moveinit__(out self, deinit existing: Self):
        self.ctx = existing.ctx^
        self.dev_a = existing.dev_a^
        self.dev_b = existing.dev_b^
        self.host = existing.host^
        self.grid_size = existing.grid_size
        self.width = existing.width
        self.height = existing.height
        self.depth = existing.depth
        self.stride = existing.stride
        self.layer_stride = existing.layer_stride
        self.gpu_active = existing.gpu_active
        self.needs_upload = existing.needs_upload
    
    fn upload_from_cpu(
        mut self,
        cpu_ptr: UnsafePointer[UInt8],
        cpu_active: Int,
    ) raises:
        for i in range(self.grid_size):
            self.host[i] = cpu_ptr[i]
        
        if cpu_active == 0:
            self.ctx.enqueue_copy(self.dev_a, self.host)
        else:
            self.ctx.enqueue_copy(self.dev_b, self.host)
        self.ctx.synchronize()
        
        self.gpu_active = cpu_active
        self.needs_upload = False
    
    fn swap_active(mut self):
        """Swap the active buffer after a kernel step."""
        self.gpu_active = 1 - self.gpu_active
    
    fn download_to_cpu(
        mut self,
        mut dst: List[UInt8],
    ) raises:
        if self.gpu_active == 0:
            self.ctx.enqueue_copy(self.host, self.dev_a)
        else:
            self.ctx.enqueue_copy(self.host, self.dev_b)
        self.ctx.synchronize()
        
        for i in range(self.grid_size):
            dst[i] = self.host[i]
    
    fn mark_dirty(mut self):
        self.needs_upload = True

