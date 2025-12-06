"""Base renderer configuration and display initialization (shared wrapper)."""

from shared.common import WIDTH, HEIGHT
from shared.renderer.base import RendererConfig, init_display as shared_init_display

# Display configuration for 1440p fullscreen
alias DISPLAY_WIDTH: Int = 2560
alias DISPLAY_HEIGHT: Int = 1440


fn init_display(title: String) raises -> RendererConfig:
    """Initialize pygame display using shared renderer utilities."""
    return shared_init_display(
        title,
        DISPLAY_WIDTH,
        DISPLAY_HEIGHT,
        fullscreen=True,
        noframe=False,
        grid_width=WIDTH,
        grid_height=HEIGHT,
    )
