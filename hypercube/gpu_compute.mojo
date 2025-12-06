"""Persistent GPU compute handler for the 4D hypercube automaton."""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor
from memory import UnsafePointer
from hypercube.gpu_kernels import hypercube_generation_kernel, get_kernel_dims, grid_layout
from hypercube.grid import HYPER_WIDTH, HYPER_HEIGHT, HYPER_DEPTH, HYPER_W

# Layout for GPU tensors (fixed-size hypercube volume)
alias grid_size = grid_layout.size


struct GPUCompute:
    """Persistent GPU compute resources for the 4D automaton."""
    var ctx: DeviceContext
    var dev_grid_a: DeviceBuffer[DType.uint8]
    var dev_grid_b: DeviceBuffer[DType.uint8]
    var host_buffer: HostBuffer[DType.uint8]
    var grid_size: Int
    var width: Int
    var height: Int
    var depth: Int
    var w_dim: Int
    var stride: Int
    var layer_stride: Int
    var hyperlayer_stride: Int
    var gpu_active: Int
    var needs_upload: Bool
    
    fn __init__(
        out self,
        width: Int,
        height: Int,
        depth: Int,
        w_dim: Int,
        stride: Int,
        layer_stride: Int,
        hyperlayer_stride: Int,
    ) raises:
        self.width = width
        self.height = height
        self.depth = depth
        self.w_dim = w_dim
        self.stride = stride
        self.layer_stride = layer_stride
        self.hyperlayer_stride = hyperlayer_stride
        self.grid_size = hyperlayer_stride * w_dim
        self.gpu_active = 0
        self.needs_upload = True
        
        self.ctx = DeviceContext()
        self.dev_grid_a = self.ctx.enqueue_create_buffer[DType.uint8](self.grid_size)
        self.dev_grid_b = self.ctx.enqueue_create_buffer[DType.uint8](self.grid_size)
        self.host_buffer = self.ctx.enqueue_create_host_buffer[DType.uint8](self.grid_size)
        self.ctx.synchronize()
    
    fn __moveinit__(out self, deinit existing: Self):
        self.ctx = existing.ctx^
        self.dev_grid_a = existing.dev_grid_a^
        self.dev_grid_b = existing.dev_grid_b^
        self.host_buffer = existing.host_buffer^
        self.grid_size = existing.grid_size
        self.width = existing.width
        self.height = existing.height
        self.depth = existing.depth
        self.w_dim = existing.w_dim
        self.stride = existing.stride
        self.layer_stride = existing.layer_stride
        self.hyperlayer_stride = existing.hyperlayer_stride
        self.gpu_active = existing.gpu_active
        self.needs_upload = existing.needs_upload
    
    fn upload_from_cpu(
        mut self,
        cpu_ptr: UnsafePointer[UInt8],
        cpu_active: Int,
    ) raises:
        """Upload CPU grid state to GPU (sync active buffer)."""
        for i in range(self.grid_size):
            self.host_buffer[i] = cpu_ptr[i]
        
        if cpu_active == 0:
            self.ctx.enqueue_copy(self.dev_grid_a, self.host_buffer)
        else:
            self.ctx.enqueue_copy(self.dev_grid_b, self.host_buffer)
        self.ctx.synchronize()
        
        self.gpu_active = cpu_active
        self.needs_upload = False
    
    fn step(mut self) raises:
        """Compute one generation on GPU."""
        var dims = get_kernel_dims(self.width, self.height, self.depth, self.w_dim)
        
        if self.gpu_active == 0:
            var read_tensor = LayoutTensor[DType.uint8, grid_layout](self.dev_grid_a)
            var write_tensor = LayoutTensor[DType.uint8, grid_layout](self.dev_grid_b)
            
            self.ctx.enqueue_function_checked[hypercube_generation_kernel, hypercube_generation_kernel](
                read_tensor,
                write_tensor,
                self.width,
                self.height,
                self.depth,
                self.w_dim,
                self.stride,
                self.layer_stride,
                self.hyperlayer_stride,
                grid_dim=(dims.grid_x, dims.grid_y, dims.grid_z),
                block_dim=(dims.block_x, dims.block_y, dims.block_z),
            )
        else:
            var read_tensor = LayoutTensor[DType.uint8, grid_layout](self.dev_grid_b)
            var write_tensor = LayoutTensor[DType.uint8, grid_layout](self.dev_grid_a)
            
            self.ctx.enqueue_function_checked[hypercube_generation_kernel, hypercube_generation_kernel](
                read_tensor,
                write_tensor,
                self.width,
                self.height,
                self.depth,
                self.w_dim,
                self.stride,
                self.layer_stride,
                self.hyperlayer_stride,
                grid_dim=(dims.grid_x, dims.grid_y, dims.grid_z),
                block_dim=(dims.block_x, dims.block_y, dims.block_z),
            )
        
        self.gpu_active = 1 - self.gpu_active
    
    fn download_to_cpu(
        mut self,
        mut dst: List[UInt8],
    ) raises:
        """Download current GPU state to CPU list buffer."""
        if self.gpu_active == 0:
            self.ctx.enqueue_copy(self.host_buffer, self.dev_grid_a)
        else:
            self.ctx.enqueue_copy(self.host_buffer, self.dev_grid_b)
        self.ctx.synchronize()
        
        for i in range(self.grid_size):
            dst[i] = self.host_buffer[i]
    
    fn mark_dirty(mut self):
        """Mark CPU state as changed; next step will upload."""
        self.needs_upload = True


