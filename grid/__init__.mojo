# Grid Game of Life package
#
# A 2D cellular automaton with simultaneous cell updates based on neighbor counts.
#
# Rules:
# - Any live cell with 2-3 live neighbors survives
# - Any dead cell with exactly 3 live neighbors becomes alive
# - All other cells die or stay dead
#
# Architecture:
# - grid.mojo: Pure data container (Grid struct)
# - rules.mojo: Grid rules logic
# - cpu_compute.mojo: CPU SIMD computation (CPUCompute)
# - gpu_compute.mojo: GPU computation (GPUCompute)
# - gpu_kernels.mojo: GPU kernel functions
#
# Run: pixi run mojo grid/main.mojo

from .grid import Grid, SCREEN_WIDTH, SCREEN_HEIGHT, INITIAL_DENSITY
from .rules import apply_grid_rules, should_be_alive
from .cpu_compute import CPUCompute
from .gpu_kernels import grid_generation_kernel, get_kernel_dims, KernelDims
from .gpu_compute import GPUCompute
from .benchmark import BenchmarkStats, BenchmarkResult, print_benchmark_summary, save_benchmark_results
from .renderer import RendererConfig, init_display, PythonModuleRenderer
