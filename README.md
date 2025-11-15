# Cellatons - Cellular Automatons Generator (Mojo)

A cellular automatons generator in [Mojo](https://www.modular.com/mojo), showcasing both CPU and GPU execution paths for large-scale 2D rule-based automata.

## Features

- **Modular Mojo implementation**: grid logic, rule definitions, rendering, and timing utilities split into dedicated files.
- **Multiple execution modes**:
  - Sequential CPU
  - Parallel cells on CPU
  - Parallel grids on CPU
  - CuPy GPU path
- **PNG output** (optional) via Python’s Pillow integration.
- Extensive timing logs (total vs. GPU-only).

## Requirements

- WSL Fedora (or Linux) environment
- [Pixi](https://prefix.dev/docs/pixi/overview) for dependency management
- Mojo 0.25+ (MAX-nightly channel)
- Python with `cupy` (CUDA-enabled GPU recommended)
- Pillow (for PNG rendering)

All dependencies are pinned in `pixi.toml` / `pixi.lock`.

## Repository Structure

```
.
├── grid.mojo               # Grid struct, CPU and CuPy GPU logic
├── rule.mojo               # Rule definition
├── rule_container.mojo     # Pre-defined rule set
├── renderer.mojo           # PNG export via Pillow
├── gpu_timing_result.mojo  # Lightweight struct for GPU timing stats
├── common.mojo             # Global aliases + helpers
├── main.mojo               # Entry point, orchestrates runs + logging
├── pixi.toml / pixi.lock   # Pixi env definition
├── generated/              # PNG outputs (ignored in Git)
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
3. **Run benchmarks**:
   ```bash
   pixi run mojo main.mojo
   ```

### Disabling PNG Output
By default, PNG generation is enabled. Attention: PNG rendering is expensive (hundreds of seconds on huge grids).

To disable, in `common.mojo`, set:
```mojo
alias RENDER_PNGS: Bool = False
```

### Adjusting Grid Size

`common.mojo` exposes:
```mojo
alias WIDTH: Int = …
alias HEIGHT: Int = …
alias CELL_SIZE: Int = …
```
Larger grids (e.g., 20k × 10k) keep the GPU busy; smaller grids generate patterns faster.
