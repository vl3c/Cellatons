"""Base renderer configuration and display initialization."""

from python import Python, PythonObject
from shared.common import WIDTH, HEIGHT

# Display configuration for 1440p fullscreen
alias DISPLAY_WIDTH: Int = 2560
alias DISPLAY_HEIGHT: Int = 1440


struct RendererConfig(Movable):
    """Runtime display configuration for pygame rendering."""
    var display_width: Int
    var display_height: Int
    var view_left: Int
    var screen: PythonObject
    var pygame: PythonObject
    var clock: PythonObject
    
    fn __init__(
        out self,
        pygame: PythonObject,
        screen: PythonObject,
        display_width: Int,
        display_height: Int,
    ) raises:
        self.pygame = pygame
        self.screen = screen
        self.display_width = display_width
        self.display_height = display_height
        self.view_left = (WIDTH - display_width) // 2
        self.clock = pygame.time.Clock()
    
    fn __moveinit__(out self, deinit existing: Self):
        self.display_width = existing.display_width
        self.display_height = existing.display_height
        self.view_left = existing.view_left
        self.screen = existing.screen
        self.pygame = existing.pygame
        self.clock = existing.clock
    
    fn max_scroll_rows(self) -> Int:
        """Calculate maximum scroll position (total rows - visible rows)."""
        return HEIGHT - self.display_height


fn init_display(title: String) raises -> RendererConfig:
    """Initialize pygame display in fullscreen mode with vsync.
    
    Args:
        title: Window title (shown in taskbar)
    
    Returns:
        RendererConfig with runtime dimensions and pygame objects.
    """
    # Set SDL environment variables for vsync before pygame.init()
    var os = Python.import_module("os")
    os.environ["SDL_RENDER_VSYNC"] = "1"
    os.environ["SDL_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR"] = "0"
    
    var pygame = Python.import_module("pygame")
    pygame.init()
    
    # Use fixed 1440p dimensions for fullscreen
    var display_width = DISPLAY_WIDTH
    var display_height = DISPLAY_HEIGHT
    
    # Clamp to grid width if needed
    if display_width > WIDTH:
        display_width = WIDTH
    
    # Create fullscreen window with hardware acceleration + double buffering
    var flags = pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF
    var screen = pygame.display.set_mode(
        Python.tuple(display_width, display_height),
        flags
    )
    pygame.display.set_caption(title)
    
    return RendererConfig(pygame, screen, display_width, display_height)
