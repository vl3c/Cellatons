"""Python-side renderer for the 3D cube automaton.

Renders a rotating cube cube with:
- Black background
- White live cubes
- Transparent, faded gray grid lines
"""

import math
from typing import Optional
from ctypes import c_uint8, POINTER, cast

import numpy as np

BG_COLOR = (0, 0, 0)
cube_COLOR = (255, 255, 255)
GRID_COLOR = (180, 180, 180, 90)  # slightly stronger alpha for visibility
STATUS_COLOR = (200, 200, 200)
FONT_NAME = "sourcecodepro"
FONT_SIZE = 24
FONT_BOLD = True
cube_SIZE = 4  # side length in pixels
GRID_STEP = 8   # draw a lattice line every GRID_STEP cubes
MAX_VISIBLE_cubeS = 60_000  # cap rendered cubes for performance


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


def create_grid_view(ptr: int, stride: int, height: int, width: int, depth: int) -> np.ndarray:
    """Create zero-copy numpy view over the Mojo grid buffer."""
    total_size = stride * height * depth
    c_array_type = c_uint8 * total_size
    c_ptr = cast(ptr, POINTER(c_array_type))
    return np.ctypeslib.as_array(c_ptr.contents).reshape(depth, height, stride)


def _norm_coord(v: int, size: int) -> float:
    """Map integer coord to [-0.5, 0.5] for a given axis size."""
    return (float(v) / float(size - 1)) - 0.5


def _axis_range(limit: int, step: int) -> list[int]:
    """Build axis ticks with final cell included."""
    vals = list(range(0, limit, step))
    if limit - 1 not in vals:
        vals.append(limit - 1)
    return vals


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


def _render_grid_lines(pygame, screen, lines: np.ndarray, rot: np.ndarray, display_w: int, display_h: int):
    """Draw faded grid lines onto an overlay surface."""
    overlay = pygame.Surface((display_w, display_h), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 0))
    
    scale = min(display_w, display_h) * 0.6  # same scale as cubes
    cx, cy = display_w * 0.5, display_h * 0.5
    
    for segment in lines:
        pts = segment  # (2,3)
        u, v, z = _project(pts, rot, scale, cx, cy)
        if u is None:
            continue
        if u.shape[0] != 2:
            continue
        pygame.draw.aaline(
            overlay,
            GRID_COLOR,
            (int(u[0]), int(v[0])),
            (int(u[1]), int(v[1])),
        )
    screen.blit(overlay, (0, 0))


def _extract_alive(
    grid: np.ndarray,
    width_logical: int,
    height_logical: int,
    depth_logical: int,
):
    """Get alive coords cropped to logical dimensions."""
    depth_raw, height_raw, stride = grid.shape
    depth = min(depth_raw, depth_logical)
    height = min(height_raw, height_logical)
    width = min(stride, width_logical)
    alive = np.argwhere(grid[:depth, :height, :width] > 0)
    return alive, width, height, depth


def _cap_alive(alive: np.ndarray) -> np.ndarray:
    """Cap number of rendered cubes for performance."""
    if alive.shape[0] > MAX_VISIBLE_cubeS:
        idx = np.random.choice(alive.shape[0], size=MAX_VISIBLE_cubeS, replace=False)
        return alive[idx]
    return alive


def _normalize_points(
    alive: np.ndarray,
    width_logical: int,
    height_logical: int,
    depth_logical: int,
) -> np.ndarray:
    """Convert integer coordinates to normalized cube coords [-0.5, 0.5]."""
    z = (alive[:, 0].astype(np.float32) / float(depth_logical - 1)) - 0.5
    y = (alive[:, 1].astype(np.float32) / float(height_logical - 1)) - 0.5
    x = (alive[:, 2].astype(np.float32) / float(width_logical - 1)) - 0.5
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


def _draw_cube_blocks(u: np.ndarray, v: np.ndarray, display_w: int, display_h: int):
    """Draw each cube as a small block with a gap for grid lines."""
    full_size = cube_SIZE
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
    
    _state.rgb_buffer[vv[mask], uu[mask], :] = cube_COLOR


def _render_cubes(
    grid: np.ndarray,
    rot: np.ndarray,
    display_w: int,
    display_h: int,
    width_logical: int,
    height_logical: int,
    depth_logical: int,
) -> None:
    """Render live cubes as white points into the rgb buffer."""
    _state.rgb_buffer.fill(0)
    
    alive, _, _, _ = _extract_alive(grid, width_logical, height_logical, depth_logical)
    if alive.size == 0:
        return
    
    alive = _cap_alive(alive)
    points = _normalize_points(alive, width_logical, height_logical, depth_logical)
    
    scale = min(display_w, display_h) * 0.6
    cx, cy = display_w * 0.5, display_h * 0.5
    u, v, z_depth = _project(points, rot, scale, cx, cy)
    if u is None:
        return
    
    filtered = _filter_screen(u, v, z_depth, display_w, display_h)
    if filtered is None:
        return
    u_filt, v_filt = filtered
    
    _draw_cube_blocks(u_filt, v_filt, display_w, display_h)


def _render_status(
    screen,
    display_width: int,
    generation: int,
    fps: int,
    gen_time_ms: float,
) -> None:
    text = f"Gen: {generation:,} | {fps} FPS | GPU ({gen_time_ms:.2f}ms)"
    surface = _state.font.render(text, True, STATUS_COLOR)
    rect = surface.get_rect()
    rect.topright = (display_width - 16, 12)
    screen.blit(surface, rect)


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
    width_logical: Optional[int] = None,
    height_logical: Optional[int] = None,
    depth_logical: Optional[int] = None,
) -> None:
    """Render one frame: grid lines + live cubes + status."""
    depth, height, stride = grid_np.shape
    if width_logical is None:
        width_logical = stride
    if height_logical is None:
        height_logical = height
    if depth_logical is None:
        depth_logical = depth
    
    _state.ensure_initialized(
        pygame,
        display_width,
        display_height,
        (width_logical, height_logical, depth_logical),
    )
    
    screen.fill(BG_COLOR)
    
    if not paused:
        _state.angle += 0.005
    yaw = _state.angle * 0.7
    pitch = _state.angle * 0.45
    rot = _rotation_matrix(yaw, pitch)
    
    # Render cubes into rgb_buffer and blit FIRST
    width_logical, height_logical, depth_logical = _state.grid_dims
    
    _render_cubes(
        grid_np,
        rot,
        display_width,
        display_height,
        width_logical,
        height_logical,
        depth_logical,
    )
    pygame.surfarray.blit_array(_state.surface, _state.rgb_buffer.swapaxes(0, 1))
    screen.blit(_state.surface, (0, 0))
    
    # Draw grid lines ON TOP of cubes
    _render_grid_lines(pygame, screen, _state.grid_lines, rot, display_width, display_height)
    
    _render_status(screen, display_width, generation, fps, gen_time_ms)


if __name__ == "__main__":
    # Minimal sanity run (not a full interactive viewer)
    import pygame
    
    display_w, display_h = 800, 600
    pygame.init()
    screen = pygame.display.set_mode((display_w, display_h))
    
    depth, height, width = 8, 8, 8
    grid = np.zeros((depth, height, width), dtype=np.uint8)
    grid[4, 4, 4] = 1
    
    clock = pygame.time.Clock()
    for _ in range(3):
        render_frame(pygame, screen, grid, display_w, display_h, 0, 60, 0.0)
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


