"""Persistent GPU compute handler for the cube automaton."""

from gpu.host import DeviceContext, DeviceBuffer
from layout import LayoutTensor
from memory import UnsafePointer
from shared.gpu_pingpong import PingPongGPU3D
from cube.gpu_kernels import cube_generation_kernel, get_kernel_dims, grid_layout
from cube.grid import cube_WIDTH, cube_HEIGHT, cube_DEPTH


struct GPUCompute:
    """Persistent GPU compute resources for the 3D automaton."""
    var pingpong: PingPongGPU3D
    var width: Int
    var height: Int
    var depth: Int
    var stride: Int
    var layer_stride: Int
    var gpu_active: Int
    var needs_upload: Bool
    
    fn __init__(out self, width: Int, height: Int, depth: Int, stride: Int, layer_stride: Int) raises:
        self.width = width
        self.height = height
        self.depth = depth
        self.stride = stride
        self.layer_stride = layer_stride
        self.pingpong = PingPongGPU3D(width, height, depth, stride, layer_stride)
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
        """Compute one generation on GPU."""
        var dims = get_kernel_dims(self.width, self.height, self.depth)
        
        # Launch kernel based on active buffer
        if self.pingpong.gpu_active == 0:
            var read_tensor = LayoutTensor[DType.uint8, grid_layout](self.pingpong.dev_a)
            var write_tensor = LayoutTensor[DType.uint8, grid_layout](self.pingpong.dev_b)
            
            self.pingpong.ctx.enqueue_function_checked[cube_generation_kernel, cube_generation_kernel](
                read_tensor,
                write_tensor,
                self.width,
                self.height,
                self.depth,
                self.stride,
                self.layer_stride,
                grid_dim=(dims.grid_x, dims.grid_y, dims.grid_z),
                block_dim=(dims.block_x, dims.block_y, dims.block_z),
            )
        else:
            var read_tensor = LayoutTensor[DType.uint8, grid_layout](self.pingpong.dev_b)
            var write_tensor = LayoutTensor[DType.uint8, grid_layout](self.pingpong.dev_a)
            
            self.pingpong.ctx.enqueue_function_checked[cube_generation_kernel, cube_generation_kernel](
                read_tensor,
                write_tensor,
                self.width,
                self.height,
                self.depth,
                self.stride,
                self.layer_stride,
                grid_dim=(dims.grid_x, dims.grid_y, dims.grid_z),
                block_dim=(dims.block_x, dims.block_y, dims.block_z),
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

