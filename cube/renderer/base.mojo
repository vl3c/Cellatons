"""Display initialization for the cube viewer (shared wrapper)."""

from shared.renderer.base import RendererConfig, init_display as shared_init_display

# Display configuration (borderless fullscreen 1440p)
alias DISPLAY_WIDTH: Int = 2560
alias DISPLAY_HEIGHT: Int = 1440


fn init_display(title: String) raises -> RendererConfig:
    """Initialize pygame window."""
    return shared_init_display(
        title,
        DISPLAY_WIDTH,
        DISPLAY_HEIGHT,
        fullscreen=True,
        noframe=True,
        grid_width=DISPLAY_WIDTH,
        grid_height=DISPLAY_HEIGHT,
    )


