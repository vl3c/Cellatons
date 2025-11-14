from python import Python
from grid import Grid
from rule_container import RuleContainer
from common import get_filename

struct Renderer:
    fn __init__(out self):
        pass
    
    fn save_pngs(self, grids: List[Grid], rule_container: RuleContainer) raises:
        var py_time = Python.import_module("time")
        var py_builtins = Python.import_module("builtins")
        
        var start = py_time.time()
        
        for i in range(len(rule_container.rules)):
            var filename = get_filename(rule_container.rules[i].output_name)
            grids[i].save_png(filename)
            print(rule_container.rules[i].name, "saved to", filename)
        
        var end = py_time.time()
        var elapsed = py_builtins.format(end - start, ".3f")
        print("Saved", len(rule_container.rules), "PNGs sequentially on CPU in", elapsed, "seconds")

