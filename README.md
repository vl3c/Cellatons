# Cellatons — Cellular Automata in Mojo 🔥

High-performance cellular automata implementations in [Mojo](https://www.modular.com/mojo), featuring CPU (SIMD-vectorized) and native GPU execution paths. Two complete sub-projects demonstrate both 1D and 2D cellular automata with real-time visualization and comprehensive benchmarking.

## Sub-Projects

### 1. Elementary Cellular Automata (1D)
Rule-based 1D automata (Rule 30, 110, 254, etc.) with pattern evolution from a single center cell. Supports massive grids (20K×10K) for deep pattern exploration.

### 2. Conway's Game of Life (2D)
Classic 2D simultaneous-update automaton with real-time pygame visualization at 1440p. Interactive controls for pause, reset, and GPU/CPU mode switching.

---

## Features

- **Mojo-native implementation**: Grid logic, rule engines, renderers, and compute modules organized into clean packages
- **Multiple execution modes**:
  - Sequential CPU
  - Parallel cells on CPU (multi-threaded)  
  - Parallel grids on CPU
  - SIMD-vectorized CPU (AVX-512, 64-cell vectors)
  - CuPy GPU (via Python interop)
  - Native Mojo GPU kernels
- **Real-time visualization**: pygame-powered viewers with FPS counter and control overlays
- **PNG export**: Optional Pillow-based image generation for elementary CA patterns
- **Comprehensive benchmarking**: Detailed timing stats (min/max/avg/stddev) with report generation
- **Extensive test suites**: 70+ unit tests covering grid operations, rules, CPU/GPU equivalence

---

## Requirements

- Linux environment (WSL2 on Windows works great)
- [Pixi](https://pixi.sh) for dependency management
- Mojo 0.26+ (MAX nightly channel)
- CUDA-capable GPU (optional, for GPU acceleration)

### Dependencies (managed by Pixi)
| Package | Version |
|---------|---------|
| mojo    | ≥0.26.1 |
| pillow  | ≥12.0   |
| cupy    | ≥13.6   |
| pygame  | ≥2.5    |

---

## Repository Structure

```
.
├── shared/                      # Shared utilities
│   ├── common.mojo              # Global config (WIDTH, HEIGHT, flags)
│   ├── benchmark.mojo           # Benchmark suite utilities
│   ├── logger.mojo              # Timestamped logging
│   └── gpu_timing_result.mojo   # GPU timing stats struct
│
├── elementary/                  # 1D Elementary Cellular Automata
│   ├── main.mojo                # Entry point (benchmark all modes)
│   ├── grid.mojo                # Grid struct + CPU/GPU generation
│   ├── rule.mojo                # Rule definition (3-neighbor patterns)
│   ├── rule_container.mojo      # Pre-defined rules (30, 110, 254)
│   ├── gpu_kernels.mojo         # Native Mojo GPU kernels
│   ├── renderer.mojo            # PNG export via Pillow
│   ├── renderer/                # Interactive pygame viewer
│   │   ├── base.mojo            # Renderer configuration
│   │   ├── python_module.mojo   # Mojo-Python bridge
│   │   └── viewer.py            # Python-side rendering
│   └── tests/                   # 40+ unit tests
│
├── conway/                      # Conway's Game of Life
│   ├── main.mojo                # Entry point (live viewer)
│   ├── run_benchmark.mojo       # Dedicated benchmark runner
│   ├── grid.mojo                # Double-buffered 2D grid
│   ├── rules.mojo               # B3/S23 rule implementation
│   ├── cpu_compute.mojo         # SIMD-vectorized CPU compute
│   ├── gpu_compute.mojo         # Persistent GPU buffer management
│   ├── gpu_kernels.mojo         # Native Mojo GPU kernels
│   ├── renderer/                # pygame real-time renderer
│   │   ├── base.mojo            # Display initialization
│   │   ├── python_module.mojo   # Mojo-Python bridge
│   │   └── viewer.py            # Zero-copy numpy rendering
│   ├── benchmark/               # Stats & reporting
│   │   ├── stats.mojo           # Statistical calculations
│   │   ├── result.mojo          # Result aggregation
│   │   └── report.mojo          # Console/file reports
│   └── tests/                   # 70+ unit tests
│
├── generated/                   # PNG outputs
├── run_tests.mojo               # Master test runner
├── pixi.toml / pixi.lock        # Environment definition
└── README.md
```

---

## Quick Start

### 1. Install Pixi
```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

### 2. Install Dependencies
```bash
pixi install
```

### 3. Run Conway's Game of Life (Interactive)
```bash
pixi run mojo conway/main.mojo
```

**Controls:**
- `SPACE` — Pause/Resume
- `R` — Reset (new random grid)
- `G` — Toggle GPU/CPU mode
- `Q` / `ESC` — Quit

### 4. Run Elementary CA Benchmark
```bash
pixi run mojo elementary/main.mojo
```
Runs all execution modes on a 20K×10K grid and prints timing comparison.

### 5. Run Conway Benchmark
```bash
pixi run mojo conway/run_benchmark.mojo
```
Runs 10,000 generations at 4K resolution, outputs detailed report to `conway_benchmark.txt`.

---

## Running Tests

```bash
# All elementary tests + renderer tests
pixi run mojo run_tests.mojo

# Conway's Game of Life tests
pixi run mojo conway/run_tests.mojo
```

---

## Configuration

### Elementary CA (`shared/common.mojo`)
```mojo
alias WIDTH: Int = 20_000       # Grid width (cells)
alias HEIGHT: Int = 10_000      # Grid height (rows/generations)
alias PIXELS_PER_CELL: Int = 2  # PNG scale factor
alias RENDER_PNGS: Bool = False # Enable PNG generation
```

### Conway's Game of Life (`conway/grid.mojo`)
```mojo
alias SCREEN_WIDTH: Int = 2560   # Display width (1440p)
alias SCREEN_HEIGHT: Int = 1440  # Display height
alias INITIAL_DENSITY: Float64 = 0.15  # Random fill density
```

---

## Performance

Both sub-projects leverage:
- **AVX-512 SIMD**: 64-byte aligned buffers for vectorized processing
- **Double-buffering**: Ping-pong buffers for zero-copy generation swaps
- **Native GPU kernels**: Mojo's GPU programming model for direct CUDA execution
- **Zero-copy rendering**: numpy views over Mojo memory for efficient Python interop

Typical performance on RTX-class GPUs:
- Elementary CA: 200M+ cells generated in <1s
- Conway: 8M+ cells/generation at 60+ FPS (GPU mode)

---

## License

MIT
