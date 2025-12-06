"""Display initialization for the voxel viewer."""

from python import Python, PythonObject

# Display configuration (borderless fullscreen 1440p)
alias DISPLAY_WIDTH: Int = 2560
alias DISPLAY_HEIGHT: Int = 1440


struct RendererConfig(Movable):
    var display_width: Int
    var display_height: Int
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
        self.clock = pygame.time.Clock()
    
    fn __moveinit__(out self, deinit existing: Self):
        self.display_width = existing.display_width
        self.display_height = existing.display_height
        self.screen = existing.screen
        self.pygame = existing.pygame
        self.clock = existing.clock


fn init_display(title: String) raises -> RendererConfig:
    """Initialize pygame window."""
    var os = Python.import_module("os")
    os.environ["SDL_RENDER_VSYNC"] = "1"
    os.environ["SDL_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR"] = "0"
    
    var pygame = Python.import_module("pygame")
    pygame.init()
    
    var display_width = DISPLAY_WIDTH
    var display_height = DISPLAY_HEIGHT
    
    var flags = pygame.FULLSCREEN | pygame.NOFRAME | pygame.HWSURFACE | pygame.DOUBLEBUF
    var screen = pygame.display.set_mode(
        Python.tuple(display_width, display_height),
        flags
    )
    pygame.display.set_caption(title)
    
    return RendererConfig(pygame, screen, display_width, display_height)


