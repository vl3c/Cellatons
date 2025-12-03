"""Python-side fast renderer for Conway's Game of Life visualization.

This module is called from Mojo's PythonModuleRenderer.
All numpy/pygame operations happen in pure Python for maximum performance.

Unlike elementary CA, Conway renders the entire grid at once (no scrolling).
"""

import numpy as np
from ctypes import c_uint8, POINTER, cast
from typing import Optional

# UI Configuration
FONT_NAME = "sourcecodepro"  # Real monospace font (prevents text jiggle)
FONT_SIZE = 36  # Main status text
HINT_FONT_SIZE = 24  # Control hints
FONT_BOLD = True
STATUS_COLOR = (0, 255, 0)  # Green for visibility on B&W
STATUS_MARGIN = 20


class RenderState:
    """Holds pre-allocated buffers and font for rendering.
    
    Using a class instead of globals for better testability and clarity.
    """
    __slots__ = ('rgb_buffer', 'surface', 'font', 'hint_font',
                 'display_width', 'display_height')
    
    def __init__(self):
        self.rgb_buffer: Optional[np.ndarray] = None
        self.surface = None
        self.font = None
        self.hint_font = None
        self.display_width: int = 0
        self.display_height: int = 0
    
    def ensure_initialized(self, pygame, display_width: int, display_height: int) -> None:
        """Initialize or resize buffers if needed."""
        if (self.rgb_buffer is None or 
            self.display_width != display_width or 
            self.display_height != display_height):
            
            self.display_width = display_width
            self.display_height = display_height
            self.rgb_buffer = np.zeros((display_height, display_width, 3), dtype=np.uint8)
            self.surface = pygame.Surface((display_width, display_height))
            
            if self.font is None:
                pygame.font.init()
                self.font = pygame.font.SysFont(FONT_NAME, FONT_SIZE, bold=FONT_BOLD)
                self.hint_font = pygame.font.SysFont(FONT_NAME, HINT_FONT_SIZE, bold=FONT_BOLD)


# Global render state (singleton for performance)
_state = RenderState()


def create_grid_view(grid_ptr: int, stride: int, height: int, width: int) -> np.ndarray:
    """Create zero-copy numpy view over Mojo's grid buffer.
    
    Args:
        grid_ptr: Integer pointer to grid.cells buffer
        stride: Row stride (aligned width for SIMD)
        height: Grid height in rows
        width: Grid width in columns
    
    Returns:
        numpy array view of shape (height, stride)
    """
    total_size = stride * height
    c_array_type = c_uint8 * total_size
    ptr = cast(grid_ptr, POINTER(c_array_type))
    return np.ctypeslib.as_array(ptr.contents).reshape(height, stride)


def render_frame(
    pygame,
    screen,
    grid_np: np.ndarray,
    display_width: int,
    display_height: int,
    generation: int = 0,
    fps: int = 0,
    mode: str = "GPU",
    gen_time_ms: float = 0.0,
) -> None:
    """Render the full grid to pygame screen with status overlay.
    
    Args:
        pygame: pygame module reference
        screen: pygame display surface
        grid_np: numpy view of grid data (height, stride)
        display_width: Window width in pixels
        display_height: Window height in pixels
        generation: Current generation number
        fps: Current frames per second
        mode: Compute mode ("GPU" or "CPU")
        gen_time_ms: Last generation computation time in milliseconds
    """
    _state.ensure_initialized(pygame, display_width, display_height)
    
    # Extract visible region (crop stride to display width)
    visible = grid_np[:display_height, :display_width]
    
    # Scale 0/1 to 0/255 and create grayscale RGB
    scaled = (visible * 255).astype(np.uint8)
    _state.rgb_buffer[:, :, 0] = scaled
    _state.rgb_buffer[:, :, 1] = scaled
    _state.rgb_buffer[:, :, 2] = scaled
    
    # Blit grid surface to screen (swapaxes for pygame's column-major format)
    pygame.surfarray.blit_array(_state.surface, _state.rgb_buffer.swapaxes(0, 1))
    screen.blit(_state.surface, (0, 0))
    
    # Draw status overlay
    _render_status(screen, display_width, generation, fps, mode, gen_time_ms)


def _render_status(
    screen, 
    display_width: int, 
    generation: int, 
    fps: int,
    mode: str,
    gen_time_ms: float,
) -> None:
    """Render generation count, FPS, mode, and control hints."""
    # Status line
    status_text = f"Gen: {generation:,} | {fps} FPS | {mode} ({gen_time_ms:.2f}ms)"
    text_surface = _state.font.render(status_text, True, STATUS_COLOR)
    text_rect = text_surface.get_rect()
    text_rect.topright = (display_width - STATUS_MARGIN, STATUS_MARGIN)
    screen.blit(text_surface, text_rect)
    
    # Control hints
    hints = "SPACE: Pause | R: Reset | G: Toggle GPU/CPU | Q: Quit"
    hint_surface = _state.hint_font.render(hints, True, STATUS_COLOR)
    hint_rect = hint_surface.get_rect()
    hint_rect.topright = (display_width - STATUS_MARGIN, STATUS_MARGIN + FONT_SIZE + 8)
    screen.blit(hint_surface, hint_rect)


def reset_state() -> None:
    """Reset render state (useful for testing)."""
    global _state
    _state = RenderState()

