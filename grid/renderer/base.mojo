"""Base renderer configuration and display initialization for Grid Game of Life."""

from grid.grid import SCREEN_WIDTH, SCREEN_HEIGHT
from shared.renderer.base import RendererConfig, init_display as shared_init_display

# Display configuration for 1440p fullscreen (matches grid exactly)
alias DISPLAY_WIDTH: Int = SCREEN_WIDTH   # 2560
alias DISPLAY_HEIGHT: Int = SCREEN_HEIGHT  # 1440


fn init_display(title: String) raises -> RendererConfig:
    """Initialize pygame display in fullscreen mode with vsync."""
    return shared_init_display(
        title,
        DISPLAY_WIDTH,
        DISPLAY_HEIGHT,
        fullscreen=True,
        noframe=False,
        grid_width=SCREEN_WIDTH,
        grid_height=SCREEN_HEIGHT,
    )

