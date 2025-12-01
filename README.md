# Cellatons - Cellular Automata Generator (Mojo)

A cellular automata generator in [Mojo](https://www.modular.com/mojo), showcasing both CPU and GPU execution paths. This project contains two sub-projects:

1. **Elementary Cellular Automata** - 1D rule-based automata (Rule 30, 110, 254, etc.)
2. **Conway's Game of Life** - 2D simultaneous-update automaton (coming soon)

## Features

- **Modular Mojo implementation**: grid logic, rule definitions, rendering, and timing utilities split into dedicated files.
- **Multiple execution modes**:
  - Sequential CPU
  - Parallel cells on CPU
  - Parallel grids on CPU
  - CuPy GPU path
  - Native Mojo GPU path
- **PNG output** (optional) via Python's Pillow integration.
- Extensive timing logs (total vs. GPU-only).

## Requirements

- WSL Fedora (or Linux) environment
- [Pixi](https://pixi.sh) for dependency management
- Mojo 0.25+ (MAX-nightly channel)
- Python with `cupy` (CUDA-enabled GPU recommended)
- Pillow (for PNG rendering)

All dependencies are pinned in `pixi.toml` / `pixi.lock`.

## Repository Structure

```
.
├── shared/                   # Shared utilities
│   ├── common.mojo           # Global aliases + helpers (WIDTH, HEIGHT, etc.)
│   └── gpu_timing_result.mojo # Lightweight struct for GPU timing stats
├── elementary/               # 1D Elementary Cellular Automata
│   ├── main.mojo             # Entry point for 1D CA
│   ├── grid.mojo             # Grid struct, CPU and GPU logic
│   ├── rule.mojo             # Rule definition (3-neighbor patterns)
│   ├── rule_container.mojo   # Pre-defined rule set (30, 110, 254)
│   ├── gpu_kernels.mojo      # Native Mojo GPU kernels
│   └── renderer.mojo         # PNG export via Pillow
├── conway/                   # Conway's Game of Life (coming soon)
│   └── main.mojo             # Entry point for GoL
├── generated/                # PNG/video outputs (shared by all sub-projects)
├── pixi.toml / pixi.lock     # Pixi env definition
└── README.md
```

## Usage

1. **Install Pixi** (if you haven't):
   ```bash
   curl -fsSL https://pixi.sh/install.sh | sh
   ```
2. **Install dependencies**:
   ```bash
   pixi install
   ```
3. **Run Elementary Cellular Automata**:
   ```bash
   pixi run mojo elementary/main.mojo
   ```
4. **Run Conway's Game of Life** (coming soon):
   ```bash
   pixi run mojo conway/main.mojo
   ```

### Disabling PNG Output
By default, PNG generation is disabled (benchmark mode). To enable, in `shared/common.mojo`, set:
```mojo
alias RENDER_PNGS: Bool = True
```

Attention: PNG rendering is expensive (hundreds of seconds on huge grids).

### Adjusting Grid Size

`shared/common.mojo` exposes:
```mojo
alias WIDTH: Int = …
alias HEIGHT: Int = …
alias CELL_SIZE: Int = …
```
Larger grids (e.g., 20k × 10k) keep the GPU busy; smaller grids generate patterns faster.
