"""Python-side fast renderer for elementary CA visualization.

This module is called from Mojo's PythonModuleRenderer.
All numpy/pygame operations happen in pure Python for maximum performance.
"""

import numpy as np
from ctypes import c_uint8, POINTER, cast
from typing import Optional

# UI Configuration
FONT_NAME = "monospace"
FONT_SIZE = 48  # Main status text
HINT_FONT_SIZE = 28  # Control hints
FONT_BOLD = True
STATUS_COLOR = (255, 0, 0)  # Red
STATUS_MARGIN = 30


class RenderState:
    """Holds pre-allocated buffers and font for rendering.
    
    Using a class instead of globals for better testability and clarity.
    """
    __slots__ = ('display_buffer', 'rgb_buffer', 'surface', 'font', 'hint_font',
                 'display_width', 'display_height')
    
    def __init__(self):
        self.display_buffer: Optional[np.ndarray] = None
        self.rgb_buffer: Optional[np.ndarray] = None
        self.surface = None
        self.font = None
        self.hint_font = None
        self.display_width: int = 0
        self.display_height: int = 0
    
    def ensure_initialized(self, pygame, display_width: int, display_height: int) -> None:
        """Initialize or resize buffers if needed."""
        if (self.display_buffer is None or 
            self.display_width != display_width or 
            self.display_height != display_height):
            
            self.display_width = display_width
            self.display_height = display_height
            self.display_buffer = np.zeros((display_height, display_width), dtype=np.uint8)
            self.rgb_buffer = np.zeros((display_height, display_width, 3), dtype=np.uint8)
            self.surface = pygame.Surface((display_width, display_height))
            
            if self.font is None:
                pygame.font.init()
                self.font = pygame.font.SysFont(FONT_NAME, FONT_SIZE, bold=FONT_BOLD)


# Global render state (singleton for performance)
_state = RenderState()


def create_grid_view(grid_ptr: int, stride: int, height: int, _width: int) -> np.ndarray:
    """Create zero-copy numpy view over Mojo's grid buffer.
    
    Args:
        grid_ptr: Integer pointer to grid.cells buffer
        stride: Row stride (aligned width for SIMD)
        height: Grid height in rows
        _width: Grid width (unused, stride is used instead)
    
    Returns:
        numpy array view of shape (height, stride)
    """
    total_size = stride * height
    c_array_type = c_uint8 * total_size
    ptr = cast(grid_ptr, POINTER(c_array_type))
    return np.ctypeslib.as_array(ptr.contents).reshape(height, stride)


def render_window(
    pygame,
    screen,
    grid_np: np.ndarray,
    start_row: int,
    display_width: int,
    display_height: int,
    view_left: int,
    progress: int = 0,
    fps: int = 0,
) -> None:
    """Render visible window to pygame screen with status overlay.
    
    Uses pre-allocated buffers to minimize allocation and reduce flickering.
    
    Args:
        pygame: pygame module reference
        screen: pygame display surface
        grid_np: numpy view of grid data (height, stride)
        start_row: First row to display (scroll position)
        display_width: Window width in pixels
        display_height: Window height in pixels  
        view_left: Left column offset for center cropping
        progress: Playback progress percentage (0-100)
        fps: Current frames per second
    """
    _state.ensure_initialized(pygame, display_width, display_height)
    
    # Extract and render grid
    _render_grid(grid_np, start_row, display_width, display_height, view_left)
    
    # Blit grid surface to screen (swapaxes for pygame's column-major format)
    pygame.surfarray.blit_array(_state.surface, _state.rgb_buffer.swapaxes(0, 1))
    screen.blit(_state.surface, (0, 0))
    
    # Draw status overlay
    _render_status(pygame, screen, display_width, progress, fps)


def _render_grid(
    grid_np: np.ndarray,
    start_row: int,
    display_width: int,
    display_height: int,
    view_left: int,
) -> None:
    """Extract visible grid region and convert to RGB."""
    grid_height = grid_np.shape[0]
    end_row = min(start_row + display_height, grid_height)
    actual_height = end_row - start_row
    
    # Clear and copy visible region
    _state.display_buffer.fill(0)
    visible = grid_np[start_row:end_row, view_left:view_left + display_width]
    _state.display_buffer[:actual_height, :] = visible
    
    # Scale 0/1 to 0/255 and create grayscale RGB (in-place for speed)
    np.multiply(_state.display_buffer, 255, out=_state.display_buffer)
    _state.rgb_buffer[:, :, 0] = _state.display_buffer
    _state.rgb_buffer[:, :, 1] = _state.display_buffer
    _state.rgb_buffer[:, :, 2] = _state.display_buffer


def _render_status(pygame, screen, display_width: int, progress: int, fps: int) -> None:
    """Render progress, FPS, and control hints in top-right corner."""
    # Status line
    status_text = f"{progress}% | {fps} FPS"
    text_surface = _state.font.render(status_text, True, STATUS_COLOR)
    text_rect = text_surface.get_rect()
    text_rect.topright = (display_width - STATUS_MARGIN, STATUS_MARGIN)
    screen.blit(text_surface, text_rect)
    
    # Control hints
    if _state.hint_font is None:
        _state.hint_font = pygame.font.SysFont(FONT_NAME, HINT_FONT_SIZE, bold=FONT_BOLD)
    
    hints = "SPACE: Pause | R: Reset | Q: Quit"
    hint_surface = _state.hint_font.render(hints, True, STATUS_COLOR)
    hint_rect = hint_surface.get_rect()
    hint_rect.topright = (display_width - STATUS_MARGIN, STATUS_MARGIN + FONT_SIZE + 10)
    screen.blit(hint_surface, hint_rect)


def reset_state() -> None:
    """Reset render state (useful for testing)."""
    global _state
    _state = RenderState()
