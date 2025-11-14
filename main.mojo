from python import Python
from rule import Rule
from grid import Grid
from rule_container import RuleContainer
from algorithm import parallelize
from common import WIDTH, HEIGHT, get_filename
from renderer import Renderer

fn generate_sequential_grids_cpu_sequential_cells_cpu(rule_container: RuleContainer) raises -> List[Grid]:
    var py_time = Python.import_module("time")
    var py_builtins = Python.import_module("builtins")
    
    var grids = List[Grid]()
    var start = py_time.time()
    
    for rule in rule_container.rules:
        var grid = Grid(WIDTH, HEIGHT)
        grid.generate_sequential_cpu(rule)
        grids.append(grid^)
    
    var end = py_time.time()
    var elapsed = py_builtins.format(end - start, ".3f")
    print("Generated", len(rule_container.rules), "grids sequentially on CPU (sequential cells on CPU) in", elapsed, "seconds")
    
    return grids^

fn generate_sequential_grids_cpu_parallel_cells_cpu(rule_container: RuleContainer) raises -> List[Grid]:
    var py_time = Python.import_module("time")
    var py_builtins = Python.import_module("builtins")
    
    var grids = List[Grid]()
    var start = py_time.time()
    
    for rule in rule_container.rules:
        var grid = Grid(WIDTH, HEIGHT)
        grid.generate_parallel_cpu(rule)
        grids.append(grid^)
    
    var end = py_time.time()
    var elapsed = py_builtins.format(end - start, ".3f")
    print("Generated", len(rule_container.rules), "grids sequentially on CPU (parallel cells on CPU) in", elapsed, "seconds")
    
    return grids^

fn generate_parallel_grids_cpu_sequential_cells_cpu(rule_container: RuleContainer) raises -> List[Grid]:
    var py_time = Python.import_module("time")
    var py_builtins = Python.import_module("builtins")
    
    var grids = List[Grid]()
    for _ in range(len(rule_container.rules)):
        grids.append(Grid(WIDTH, HEIGHT))
    
    var start = py_time.time()
    
    @parameter
    fn generate_grid(rule_idx: Int):
        grids[rule_idx].generate_sequential_cpu(rule_container.rules[rule_idx])
    
    parallelize[generate_grid](len(rule_container.rules))
    
    var end = py_time.time()
    var elapsed = py_builtins.format(end - start, ".3f")
    print("Generated", len(rule_container.rules), "grids in parallel on CPU (sequential cells on CPU) in", elapsed, "seconds")
    
    return grids^

fn main() raises:
    var py_time = Python.import_module("time")
    var py_builtins = Python.import_module("builtins")
    
    var total_start = py_time.time()
    var rule_container = RuleContainer()
    
    var grids_seq_cpu_cells_seq_cpu = generate_sequential_grids_cpu_sequential_cells_cpu(rule_container)
    var grids_seq_cpu_cells_par_cpu = generate_sequential_grids_cpu_parallel_cells_cpu(rule_container)
    var grids_par_cpu_cells_seq_cpu = generate_parallel_grids_cpu_sequential_cells_cpu(rule_container)
    
    var renderer = Renderer()
    renderer.save_pngs(grids_par_cpu_cells_seq_cpu, rule_container)
    
    var total_end = py_time.time()
    var total_elapsed = py_builtins.format(total_end - total_start, ".3f")
    print("Total time:", total_elapsed, "seconds")
