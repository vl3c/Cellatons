"""Persistent GPU compute handler for Conway's Game of Life.

Keeps GPU buffers allocated across frames to avoid per-frame allocation overhead.
"""

from sys import has_accelerator
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor
from conway.gpu_kernels import conway_generation_kernel, get_kernel_dims
from conway.grid import SCREEN_WIDTH, SCREEN_HEIGHT

# Layout for GPU tensors
alias grid_layout = Layout.row_major(SCREEN_WIDTH * SCREEN_HEIGHT)


struct GPUCompute:
    """Persistent GPU compute resources for Conway simulation.
    
    Allocates GPU buffers once and reuses them across frames.
    Uses double-buffering on GPU side for ping-pong computation.
    """
    var ctx: DeviceContext
    var dev_grid_a: DeviceBuffer[DType.uint8]
    var dev_grid_b: DeviceBuffer[DType.uint8]
    var host_buffer: HostBuffer[DType.uint8]
    var grid_size: Int
    var width: Int
    var height: Int
    var stride: Int
    var gpu_active: Int  # Which GPU buffer is current (0=A, 1=B)
    var needs_upload: Bool  # True if CPU state needs to be uploaded
    
    fn __init__(out self, width: Int, height: Int, stride: Int) raises:
        """Initialize GPU resources.
        
        Args:
            width: Grid width
            height: Grid height
            stride: Row stride (aligned for SIMD)
        """
        self.width = width
        self.height = height
        self.stride = stride
        self.grid_size = stride * height
        self.gpu_active = 0
        self.needs_upload = True
        
        # Create device context
        self.ctx = DeviceContext()
        
        # Allocate persistent device buffers
        self.dev_grid_a = self.ctx.enqueue_create_buffer[DType.uint8](self.grid_size)
        self.dev_grid_b = self.ctx.enqueue_create_buffer[DType.uint8](self.grid_size)
        self.host_buffer = self.ctx.enqueue_create_host_buffer[DType.uint8](self.grid_size)
        self.ctx.synchronize()
    
    fn upload_from_cpu(
        mut self,
        cpu_ptr: UnsafePointer[UInt8],
        cpu_active: Int,
    ) raises:
        """Upload CPU grid state to GPU.
        
        Args:
            cpu_ptr: Pointer to CPU buffer to upload
            cpu_active: Which CPU buffer is active (to sync GPU active)
        """
        # Copy to host buffer
        for i in range(self.grid_size):
            self.host_buffer[i] = cpu_ptr[i]
        
        # Upload to appropriate GPU buffer
        if cpu_active == 0:
            self.ctx.enqueue_copy(self.dev_grid_a, self.host_buffer)
        else:
            self.ctx.enqueue_copy(self.dev_grid_b, self.host_buffer)
        self.ctx.synchronize()
        
        self.gpu_active = cpu_active
        self.needs_upload = False
    
    fn step(mut self) raises:
        """Compute one generation on GPU using persistent buffers."""
        var dims = get_kernel_dims(self.width, self.height)
        
        # Launch kernel based on active buffer
        # (Avoiding MutAnyOrigin by creating tensors directly in each branch)
        if self.gpu_active == 0:
            var read_tensor = LayoutTensor[DType.uint8, grid_layout](self.dev_grid_a)
            var write_tensor = LayoutTensor[DType.uint8, grid_layout](self.dev_grid_b)
            
            self.ctx.enqueue_function_checked[conway_generation_kernel, conway_generation_kernel](
                read_tensor,
                write_tensor,
                self.width,
                self.height,
                self.stride,
                grid_dim=(dims.grid_x, dims.grid_y),
                block_dim=(dims.block_x, dims.block_y),
            )
        else:
            var read_tensor = LayoutTensor[DType.uint8, grid_layout](self.dev_grid_b)
            var write_tensor = LayoutTensor[DType.uint8, grid_layout](self.dev_grid_a)
            
            self.ctx.enqueue_function_checked[conway_generation_kernel, conway_generation_kernel](
                read_tensor,
                write_tensor,
                self.width,
                self.height,
                self.stride,
                grid_dim=(dims.grid_x, dims.grid_y),
                block_dim=(dims.block_x, dims.block_y),
            )
        
        # Swap active buffer
        self.gpu_active = 1 - self.gpu_active
    
    fn download_to_cpu[O: MutOrigin](
        mut self,
        dst_ptr: UnsafePointer[UInt8, origin=O],
    ) raises:
        """Download current GPU state to CPU buffer.
        
        Args:
            dst_ptr: Pointer to CPU buffer to write to
        """
        # Copy from current active GPU buffer to host
        if self.gpu_active == 0:
            self.ctx.enqueue_copy(self.host_buffer, self.dev_grid_a)
        else:
            self.ctx.enqueue_copy(self.host_buffer, self.dev_grid_b)
        self.ctx.synchronize()
        
        # Copy to CPU
        for i in range(self.grid_size):
            dst_ptr[i] = self.host_buffer[i]
    
    fn mark_dirty(mut self):
        """Mark that CPU state has changed and needs re-upload."""
        self.needs_upload = True
