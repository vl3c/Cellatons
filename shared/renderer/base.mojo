"""Shared renderer configuration and display initialization."""

from python import Python, PythonObject


struct RendererConfig(Movable):
    """Runtime display configuration for pygame rendering."""
    var display_width: Int
    var display_height: Int
    var view_left: Int
    var grid_height: Int
    var screen: PythonObject
    var pygame: PythonObject
    var clock: PythonObject
    
    fn __init__(
        out self,
        pygame: PythonObject,
        screen: PythonObject,
        display_width: Int,
        display_height: Int,
        view_left: Int,
        grid_height: Int,
    ) raises:
        self.pygame = pygame
        self.screen = screen
        self.display_width = display_width
        self.display_height = display_height
        self.view_left = view_left
        self.grid_height = grid_height
        self.clock = pygame.time.Clock()
    
    fn __moveinit__(out self, deinit existing: Self):
        self.display_width = existing.display_width
        self.display_height = existing.display_height
        self.view_left = existing.view_left
        self.grid_height = existing.grid_height
        self.screen = existing.screen
        self.pygame = existing.pygame
        self.clock = existing.clock
    
    fn max_scroll_rows(self) -> Int:
        """Calculate maximum scroll position (total rows - visible rows)."""
        if self.grid_height <= 0:
            return 0
        var remaining = self.grid_height - self.display_height
        if remaining < 0:
            return 0
        return remaining


fn init_display(
    title: String,
    display_width: Int,
    display_height: Int,
    fullscreen: Bool = True,
    noframe: Bool = False,
    grid_width: Int = 0,
    grid_height: Int = 0,
) raises -> RendererConfig:
    """Initialize pygame display with configurable flags.
    
    Args:
        title: Window title (shown in taskbar)
        display_width: Desired window width
        display_height: Desired window height
        fullscreen: Whether to request fullscreen
        noframe: Whether to hide the window frame
        grid_width: Logical grid width (for centering view_left)
        grid_height: Logical grid height (for scroll calculations)
    """
    # Set SDL environment variables for vsync before pygame.init()
    var os = Python.import_module("os")
    os.environ["SDL_RENDER_VSYNC"] = "1"
    os.environ["SDL_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR"] = "0"
    
    var pygame = Python.import_module("pygame")
    pygame.init()
    
    var final_width = display_width
    var final_height = display_height
    
    # Clamp display width to grid width when provided
    if grid_width > 0 and final_width > grid_width:
        final_width = grid_width
    
    # Compute view offset for wide grids (row CA)
    var view_left = 0
    if grid_width > 0 and grid_width > final_width:
        view_left = (grid_width - final_width) // 2
    
    # Build display flags
    var flags = pygame.DOUBLEBUF
    if fullscreen:
        flags |= pygame.FULLSCREEN
    if noframe:
        flags |= pygame.NOFRAME
    flags |= pygame.HWSURFACE
    
    var screen = pygame.display.set_mode(
        Python.tuple(final_width, final_height),
        flags
    )
    pygame.display.set_caption(title)
    
    return RendererConfig(
        pygame,
        screen,
        final_width,
        final_height,
        view_left,
        grid_height,
    )

