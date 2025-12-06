"""Persistent GPU compute handler for Grid Automata."""

from sys import has_accelerator
from gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor
from memory import UnsafePointer
from shared.gpu_pingpong import PingPongGPU2D
from grid.gpu_kernels import grid_generation_kernel, get_kernel_dims
from grid.grid import DISPLAY_WIDTH, DISPLAY_HEIGHT

# Layout for GPU tensors
alias grid_layout = Layout.row_major(DISPLAY_WIDTH * DISPLAY_HEIGHT)


struct GPUCompute:
    """Persistent GPU compute resources for grid simulation.
    
    Allocates GPU buffers once and reuses them across frames.
    Uses double-buffering on GPU side for ping-pong computation.
    """
    var pingpong: PingPongGPU2D
    var width: Int
    var height: Int
    var stride: Int
    var gpu_active: Int  # Which GPU buffer is current (0=A, 1=B)
    var needs_upload: Bool  # True if CPU state needs to be uploaded
    
    fn __init__(out self, width: Int, height: Int, stride: Int) raises:
        self.width = width
        self.height = height
        self.stride = stride
        self.pingpong = PingPongGPU2D(width, height, stride)
        self.gpu_active = self.pingpong.gpu_active
        self.needs_upload = self.pingpong.needs_upload
    
    fn upload_from_cpu(
        mut self,
        cpu_ptr: UnsafePointer[UInt8],
        cpu_active: Int,
    ) raises:
        self.pingpong.upload_from_cpu(cpu_ptr, cpu_active)
        self.gpu_active = self.pingpong.gpu_active
        self.needs_upload = self.pingpong.needs_upload
    
    fn step(mut self) raises:
        """Compute one generation on GPU using persistent buffers."""
        var dims = get_kernel_dims(self.width, self.height)
        
        # Launch kernel based on active buffer
        if self.pingpong.gpu_active == 0:
            var read_tensor = LayoutTensor[DType.uint8, grid_layout](self.pingpong.dev_a)
            var write_tensor = LayoutTensor[DType.uint8, grid_layout](self.pingpong.dev_b)
            
            self.pingpong.ctx.enqueue_function_checked[grid_generation_kernel, grid_generation_kernel](
                read_tensor,
                write_tensor,
                self.width,
                self.height,
                self.stride,
                grid_dim=(dims.grid_x, dims.grid_y),
                block_dim=(dims.block_x, dims.block_y),
            )
        else:
            var read_tensor = LayoutTensor[DType.uint8, grid_layout](self.pingpong.dev_b)
            var write_tensor = LayoutTensor[DType.uint8, grid_layout](self.pingpong.dev_a)
            
            self.pingpong.ctx.enqueue_function_checked[grid_generation_kernel, grid_generation_kernel](
                read_tensor,
                write_tensor,
                self.width,
                self.height,
                self.stride,
                grid_dim=(dims.grid_x, dims.grid_y),
                block_dim=(dims.block_x, dims.block_y),
            )
        
        # Swap active buffer
        self.pingpong.swap_active()
        self.gpu_active = self.pingpong.gpu_active
    
    fn download_to_cpu(
        mut self,
        mut dst: List[UInt8],
    ) raises:
        self.pingpong.download_to_cpu(dst)
    
    fn mark_dirty(mut self):
        self.pingpong.mark_dirty()
        self.needs_upload = self.pingpong.needs_upload

