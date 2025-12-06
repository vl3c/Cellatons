"""
Python-side renderer for the 4D hypercube automaton.

Modes (cycled with T):
- Slice: render a single W slice
- Max-Intensity: project max over W into 3D
- Tiled: stack several W slices along depth with gaps
- Off: skip rendering (status only)

Visuals: black background, white live voxels, transparent gray grid lines.
"""

import math
from typing import Optional
from ctypes import c_uint8, POINTER, cast

import numpy as np

# Colors and style
BG_COLOR = (0, 0, 0)
VOXEL_COLOR = (255, 255, 255)
GRID_COLOR = (180, 180, 180, 90)
STATUS_COLOR = (200, 200, 200)
FONT_NAME = "sourcecodepro"
FONT_SIZE = 24
FONT_BOLD = True
VOXEL_SIZE = 4  # side length in pixels
GRID_STEP = 16   # draw a lattice line every GRID_STEP units (coarser)
MAX_VISIBLE_VOXELS = 80_000

# Render modes
MODE_SLICE = 0
MODE_MAX = 1
MODE_TILED = 2
MODE_OFF = 3


def create_grid_view(ptr: int, stride: int, height: int, width: int, depth: int, w_dim: int) -> np.ndarray:
    """Create zero-copy numpy view over the Mojo hypercube buffer."""
    total_size = stride * height * depth * w_dim
    c_array_type = c_uint8 * total_size
    c_ptr = cast(ptr, POINTER(c_array_type))
    return np.ctypeslib.as_array(c_ptr.contents).reshape(w_dim, depth, height, stride)


class RenderState:
    __slots__ = (
        "rgb_buffer",
        "surface",
        "display_width",
        "display_height",
        "angle",
        "grid_lines",
        "grid_dims",
        "font",
    )
    
    def __init__(self):
        self.rgb_buffer: Optional[np.ndarray] = None  # (H, W, 3) uint8
        self.surface = None
        self.display_width = 0
        self.display_height = 0
        self.angle = 0.0
        self.grid_lines: Optional[np.ndarray] = None  # (N, 2, 3) float32
        self.grid_dims = (0, 0, 0)
        self.font = None
    
    def ensure_initialized(
        self,
        pygame,
        display_width: int,
        display_height: int,
        grid_dims: tuple[int, int, int],
    ) -> None:
        if (
            self.rgb_buffer is None
            or self.display_width != display_width
            or self.display_height != display_height
        ):
            self.display_width = display_width
            self.display_height = display_height
            self.rgb_buffer = np.zeros(
                (display_height, display_width, 3), dtype=np.uint8
            )
            self.surface = pygame.Surface((display_width, display_height))
            pygame.font.init()
            self.font = pygame.font.SysFont(FONT_NAME, FONT_SIZE, bold=FONT_BOLD)
        
        if self.grid_lines is None or self.grid_dims != grid_dims:
            self.grid_dims = grid_dims
            self.grid_lines = _build_grid_lines(*grid_dims)


_state = RenderState()


def reset_state() -> None:
    """Reset global renderer state (used by tests)."""
    global _state
    _state = RenderState()


# ─────────────────────────────────────────────────────────────────────────────
# Grid helpers
# ─────────────────────────────────────────────────────────────────────────────


def _norm_coord(v: int, size: int) -> float:
    """Map integer coord to [-0.5, 0.5] for a given axis size."""
    return (float(v) / float(size - 1)) - 0.5


def _axis_range(limit: int, desired_step: int) -> list[int]:
    """Build axis ticks; keep only bounds to reduce grid density."""
    if limit <= 0:
        return []
    if limit == 1:
        return [0]
    return [0, limit - 1]


def _append_lines_along_axis(
    lines: list,
    axis: int,
    iter1_vals: list[int],
    iter2_vals: list[int],
    sizes: tuple[int, int, int],
) -> None:
    """Add lines along one axis (0=x, 1=y, 2=z), iterating over the other two."""
    width, height, depth = sizes
    for v1 in iter1_vals:
        for v2 in iter2_vals:
            if axis == 0:  # X-axis lines (iterate y, z)
                p0 = (_norm_coord(0, width), _norm_coord(v1, height), _norm_coord(v2, depth))
                p1 = (_norm_coord(width - 1, width), _norm_coord(v1, height), _norm_coord(v2, depth))
            elif axis == 1:  # Y-axis lines (iterate x, z)
                p0 = (_norm_coord(v1, width), _norm_coord(0, height), _norm_coord(v2, depth))
                p1 = (_norm_coord(v1, width), _norm_coord(height - 1, height), _norm_coord(v2, depth))
            else:  # Z-axis lines (iterate x, y)
                p0 = (_norm_coord(v1, width), _norm_coord(v2, height), _norm_coord(0, depth))
                p1 = (_norm_coord(v1, width), _norm_coord(v2, height), _norm_coord(depth - 1, depth))
            lines.append((p0, p1))


def _build_axis_lines(
    xs: list[int],
    ys: list[int],
    zs: list[int],
    width: int,
    height: int,
    depth: int,
) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    lines = []
    sizes = (width, height, depth)
    _append_lines_along_axis(lines, 0, ys, zs, sizes)  # X-axis
    _append_lines_along_axis(lines, 1, xs, zs, sizes)  # Y-axis
    _append_lines_along_axis(lines, 2, xs, ys, sizes)  # Z-axis
    return lines


def _build_grid_lines(width: int, height: int, depth: int) -> np.ndarray:
    """Precompute grid line endpoints in normalized cube coords [-0.5, 0.5]."""
    step = max(1, GRID_STEP)
    xs = _axis_range(width, step)
    ys = _axis_range(height, step)
    zs = _axis_range(depth, step)
    
    lines = _build_axis_lines(xs, ys, zs, width, height, depth)
    return np.array(lines, dtype=np.float32)  # (N, 2, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Math helpers
# ─────────────────────────────────────────────────────────────────────────────


def _rotation_matrix(yaw: float, pitch: float) -> np.ndarray:
    """Build rotation matrix (yaw then pitch)."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    
    rot_yaw = np.array(
        [[cy, 0.0, sy],
         [0.0, 1.0, 0.0],
         [-sy, 0.0, cy]],
        dtype=np.float32,
    )
    rot_pitch = np.array(
        [[1.0, 0.0, 0.0],
         [0.0, cp, -sp],
         [0.0, sp, cp]],
        dtype=np.float32,
    )
    return rot_pitch @ rot_yaw


def _project(points: np.ndarray, rot: np.ndarray, scale: float, cx: float, cy: float):
    """Rotate and project 3D points to 2D screen coordinates."""
    rotated = points @ rot.T
    z = rotated[:, 2] + 1.5  # push cube forward
    valid = z > 0.05
    if not np.any(valid):
        return None, None, None
    rotated = rotated[valid]
    z = z[valid]
    x_proj = rotated[:, 0] / z
    y_proj = rotated[:, 1] / z
    
    u = (x_proj * scale + cx).astype(np.int32)
    v = (y_proj * scale + cy).astype(np.int32)
    return u, v, z


# ─────────────────────────────────────────────────────────────────────────────
# Volume extraction
# ─────────────────────────────────────────────────────────────────────────────


def _crop_volume(grid4d: np.ndarray, width: int, height: int, depth: int) -> np.ndarray:
    """Crop stride/padding to logical dimensions."""
    return grid4d[:, :depth, :height, :width]


def _tiled_volume(grid4d: np.ndarray, width: int, height: int, depth: int) -> tuple[np.ndarray, tuple[int, int, int], str]:
    """Stack several W slices along depth with a small gap."""
    w_dim = grid4d.shape[0]
    tile_count = min(4, w_dim)
    if tile_count <= 0:
        return grid4d[0, :depth, :height, :width], (width, height, depth), "Slice W0"
    
    indices = np.linspace(0, w_dim - 1, tile_count).round().astype(int)
    gap = 2
    depth_out = depth * tile_count + gap * (tile_count - 1)
    volume = np.zeros((depth_out, height, width), dtype=np.uint8)
    
    for i, idx in enumerate(indices):
        start = i * (depth + gap)
        end = start + depth
        volume[start:end, :, :] = grid4d[idx, :depth, :height, :width]
    
    label = "Tiled W slices " + ",".join(str(i) for i in indices)
    return volume, (width, height, depth_out), label


def _prepare_volume(
    grid4d: np.ndarray,
    render_mode: int,
    slice_index: int,
    width_logical: int,
    height_logical: int,
    depth_logical: int,
) -> tuple[Optional[np.ndarray], tuple[int, int, int], str, int]:
    """Derive a 3D volume from the 4D grid based on render mode."""
    cropped = _crop_volume(grid4d, width_logical, height_logical, depth_logical)
    w_dim = cropped.shape[0]
    
    if render_mode == MODE_OFF:
        return None, (width_logical, height_logical, depth_logical), "Off", w_dim
    
    if render_mode == MODE_MAX:
        volume = np.max(cropped, axis=0)
        return volume, (width_logical, height_logical, depth_logical), "Max over W", w_dim
    
    if render_mode == MODE_TILED:
        volume, dims, label = _tiled_volume(cropped, width_logical, height_logical, depth_logical)
        return volume, dims, label, w_dim
    
    # Default: slice
    w_idx = slice_index % max(1, w_dim)
    volume = cropped[w_idx, :depth_logical, :height_logical, :width_logical]
    label = f"Slice W {w_idx}/{w_dim - 1}"
    return volume, (width_logical, height_logical, depth_logical), label, w_dim


# ─────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ─────────────────────────────────────────────────────────────────────────────


def _render_grid_lines(pygame, screen, lines: np.ndarray, rot: np.ndarray, display_w: int, display_h: int):
    """Draw faded grid lines onto an overlay surface."""
    overlay = pygame.Surface((display_w, display_h), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 0))
    
    scale = min(display_w, display_h) * 0.6
    cx, cy = display_w * 0.5, display_h * 0.5
    
    for segment in lines:
        pts = segment  # (2,3)
        u, v, z = _project(pts, rot, scale, cx, cy)
        if u is None or u.shape[0] != 2:
            continue
        pygame.draw.aaline(
            overlay,
            GRID_COLOR,
            (int(u[0]), int(v[0])),
            (int(u[1]), int(v[1])),
        )
    screen.blit(overlay, (0, 0))


def _extract_alive(volume: np.ndarray) -> np.ndarray:
    """Get alive coords."""
    return np.argwhere(volume > 0)


def _cap_alive(alive: np.ndarray) -> np.ndarray:
    """Cap number of rendered voxels for performance."""
    if alive.shape[0] > MAX_VISIBLE_VOXELS:
        idx = np.random.choice(alive.shape[0], size=MAX_VISIBLE_VOXELS, replace=False)
        return alive[idx]
    return alive


def _normalize_points(
    alive: np.ndarray,
    dims: tuple[int, int, int],
) -> np.ndarray:
    """Convert integer coordinates to normalized cube coords [-0.5, 0.5]."""
    depth, height, width = dims
    z = (alive[:, 0].astype(np.float32) / float(max(1, depth - 1))) - 0.5
    y = (alive[:, 1].astype(np.float32) / float(max(1, height - 1))) - 0.5
    x = (alive[:, 2].astype(np.float32) / float(max(1, width - 1))) - 0.5
    return np.stack([x, y, z], axis=1)


def _filter_screen(
    u: np.ndarray,
    v: np.ndarray,
    z_depth: np.ndarray,
    display_w: int,
    display_h: int,
):
    """Keep projected points that are on-screen and in front of camera."""
    valid = (
        (u >= 0) & (u < display_w) &
        (v >= 0) & (v < display_h) &
        (z_depth > 0)
    )
    if not np.any(valid):
        return None, None
    return u[valid], v[valid]


def _draw_voxel_blocks(u: np.ndarray, v: np.ndarray, display_w: int, display_h: int):
    """Draw each voxel as a small block with a gap for grid lines."""
    full_size = VOXEL_SIZE
    draw_size = max(1, full_size - 1)
    half = draw_size // 2
    offsets = np.arange(draw_size, dtype=np.int32)
    dx2, dy2 = np.meshgrid(offsets, offsets, indexing="xy")  # (draw_size,draw_size)
    
    uu = u[:, None, None] + dx2[None, :, :] - half   # (N,draw_size,draw_size)
    vv = v[:, None, None] + dy2[None, :, :] - half   # (N,draw_size,draw_size)
    
    mask = (
        (uu >= 0) & (uu < display_w) &
        (vv >= 0) & (vv < display_h)
    )
    if not np.any(mask):
        return
    
    _state.rgb_buffer[vv[mask], uu[mask], :] = VOXEL_COLOR


def _render_voxels(
    volume: np.ndarray,
    rot: np.ndarray,
    display_w: int,
    display_h: int,
    dims: tuple[int, int, int],
) -> None:
    """Render live voxels as white points into the rgb buffer."""
    _state.rgb_buffer.fill(0)
    
    alive = _extract_alive(volume)
    if alive.size == 0:
        return
    
    alive = _cap_alive(alive)
    points = _normalize_points(alive, dims)
    
    scale = min(display_w, display_h) * 0.6
    cx, cy = display_w * 0.5, display_h * 0.5
    u, v, z_depth = _project(points, rot, scale, cx, cy)
    if u is None:
        return
    
    filtered = _filter_screen(u, v, z_depth, display_w, display_h)
    if filtered is None:
        return
    u_filt, v_filt = filtered
    
    _draw_voxel_blocks(u_filt, v_filt, display_w, display_h)


def _render_status(
    screen,
    display_width: int,
    generation: int,
    fps: int,
    gen_time_ms: float,
    mode_label: str,
    slice_label: str,
) -> None:
    text = f"Gen: {generation:,} | {fps} FPS | GPU ({gen_time_ms:.2f}ms) | Mode: {mode_label} | {slice_label}"
    surface = _state.font.render(text, True, STATUS_COLOR)
    rect = surface.get_rect()
    rect.topright = (display_width - 16, 12)
    screen.blit(surface, rect)


def _render_legend(screen, pygame, display_width: int, display_height: int, mode_label: str) -> None:
    """Draw a small legend with controls and current mode."""
    lines = [
        f"Mode: {mode_label}",
        "SPACE: Pause/Resume",
        "R: Reset",
        "T: Cycle mode (Slice/Max/Tiled/Off)",
        "[ ]: Move W slice (Slice mode)",
        "Q/ESC: Quit",
    ]
    y = 12
    x = 16
    for line in lines:
        surface = _state.font.render(line, True, STATUS_COLOR)
        screen.blit(surface, (x, y))
        y += surface.get_height() + 4


# ─────────────────────────────────────────────────────────────────────────────
# Public entry
# ─────────────────────────────────────────────────────────────────────────────


def render_frame(
    pygame,
    screen,
    grid_np: np.ndarray,
    display_width: int,
    display_height: int,
    generation: int = 0,
    fps: int = 0,
    gen_time_ms: float = 0.0,
    paused: bool = False,
    render_mode: int = MODE_SLICE,
    slice_index: int = 0,
    width_logical: int = 0,
    height_logical: int = 0,
    depth_logical: int = 0,
    w_dim: int = 0,
) -> None:
    """Render one frame: grid lines + live voxels + status."""
    if width_logical == 0 or height_logical == 0 or depth_logical == 0 or w_dim == 0:
        w_dim, depth_logical, height_logical, width_logical = grid_np.shape
    
    volume, grid_dims, label, w_dim_actual = _prepare_volume(
        grid_np,
        render_mode,
        slice_index,
        width_logical,
        height_logical,
        depth_logical,
    )
    
    _state.ensure_initialized(
        pygame,
        display_width,
        display_height,
        grid_dims,
    )
    
    screen.fill(BG_COLOR)
    
    if render_mode != MODE_OFF and volume is not None:
        if not paused:
            _state.angle += 0.005
        yaw = _state.angle * 0.7
        pitch = _state.angle * 0.45
        rot = _rotation_matrix(yaw, pitch)
        
        # Render voxels into rgb_buffer and blit first
        _render_voxels(
            volume,
            rot,
            display_width,
            display_height,
            grid_dims[::-1],  # dims were (width, height, depth)
        )
        pygame.surfarray.blit_array(_state.surface, _state.rgb_buffer.swapaxes(0, 1))
        screen.blit(_state.surface, (0, 0))
        
        # Draw grid lines on top
        _render_grid_lines(pygame, screen, _state.grid_lines, rot, display_width, display_height)
    
    mode_label = {
        MODE_SLICE: "Slice",
        MODE_MAX: "Max-Intensity",
        MODE_TILED: "Tiled",
        MODE_OFF: "Off",
    }.get(render_mode, "Slice")
    slice_label = label
    
    _render_status(screen, display_width, generation, fps, gen_time_ms, mode_label, slice_label)
    _render_legend(screen, pygame, display_width, display_height, mode_label)


if __name__ == "__main__":
    # Minimal sanity run (not a full interactive viewer)
    import pygame
    
    display_w, display_h = 800, 600
    pygame.init()
    screen = pygame.display.set_mode((display_w, display_h))
    
    w_dim, depth, height, width = 2, 4, 4, 4
    grid = np.zeros((w_dim, depth, height, width), dtype=np.uint8)
    grid[0, 1, 1, 1] = 1
    grid[1, 2, 2, 2] = 1
    
    clock = pygame.time.Clock()
    for _ in range(3):
        render_frame(
            pygame,
            screen,
            grid,
            display_w,
            display_h,
            0,
            60,
            0.0,
            False,
            MODE_SLICE,
            0,
            width,
            height,
            depth,
            w_dim,
        )
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


