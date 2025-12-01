from python import Python
from elementary.grid import Grid
from elementary.rule_container import RuleContainer
from shared.common import get_filename, PIXELS_PER_CELL

struct Renderer:
    fn __init__(out self):
        pass
    
    fn save_png(self, grid: Grid, filename: String) raises:
        var PIL = Python.import_module("PIL.Image")
        var builtins = Python.import_module("builtins")
        
        var width = grid.get_width()
        var height = grid.get_height()
        
        var img_width = width * PIXELS_PER_CELL
        var img_height = height * PIXELS_PER_CELL
        
        var size = builtins.tuple([img_width, img_height])
        var img = PIL.new("RGB", size, "white")
        var pixels = img.load()
        
        for row in range(height):
            for col in range(width):
                if grid.get_cell(row, col) == 1:
                    for py in range(PIXELS_PER_CELL):
                        for px in range(PIXELS_PER_CELL):
                            var x = col * PIXELS_PER_CELL + px
                            var y = row * PIXELS_PER_CELL + py
                            var black = builtins.tuple([0, 0, 0])
                            pixels[x, y] = black
        
        img.save(filename)
    
    fn save_pngs(self, grids: List[Grid], rule_container: RuleContainer) raises:
        var py_time = Python.import_module("time")
        var py_builtins = Python.import_module("builtins")
        
        var start = py_time.time()
        
        for i in range(len(rule_container.rules)):
            var filename = get_filename(rule_container.rules[i].output_name)
            self.save_png(grids[i], filename)
            print(rule_container.rules[i].name, "saved to", filename)
        
        var end = py_time.time()
        var elapsed = py_builtins.format(end - start, ".3f")
        print("Saved", len(rule_container.rules), "PNGs sequentially on CPU in", elapsed, "seconds")

