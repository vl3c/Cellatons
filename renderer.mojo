from python import Python
from grid import Grid
from rule_container import RuleContainer
from common import get_filename, CELL_SIZE

struct Renderer:
    fn __init__(out self):
        pass
    
    fn save_png(self, grid: Grid, filename: String) raises:
        var PIL = Python.import_module("PIL.Image")
        var builtins = Python.import_module("builtins")
        
        var width = grid.get_width()
        var height = grid.get_height()
        
        var img_width = width * CELL_SIZE
        var img_height = height * CELL_SIZE
        
        var size = builtins.tuple([img_width, img_height])
        var img = PIL.new("RGB", size, "white")
        var pixels = img.load()
        
        for row in range(height):
            for col in range(width):
                if grid.get_cell(row, col) == 1:
                    for py in range(CELL_SIZE):
                        for px in range(CELL_SIZE):
                            var x = col * CELL_SIZE + px
                            var y = row * CELL_SIZE + py
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

