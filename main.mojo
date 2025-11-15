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

fn generate_sequential_grids_cpu_parallel_cells_gpu(rule_container: RuleContainer) raises -> List[Grid]:
    var py_time = Python.import_module("time")
    var py_builtins = Python.import_module("builtins")
    var py_operator = Python.import_module("operator")
    
    var grids = List[Grid]()
    var total_prep = py_builtins.float(0.0)
    var total_compute = py_builtins.float(0.0)
    var total_transfer = py_builtins.float(0.0)
    var total_total = py_builtins.float(0.0)
    var total_runs = 0
    var start = py_time.time()
    
    for rule in rule_container.rules:
        var grid = Grid(WIDTH, HEIGHT)
        var stats = grid.generate_parallel_cells_cupy_gpu(rule)
        total_prep = py_operator.add(total_prep, stats.prep)
        total_compute = py_operator.add(total_compute, stats.compute)
        total_transfer = py_operator.add(total_transfer, stats.transfer)
        total_total = py_operator.add(total_total, stats.total)
        total_runs += stats.runs
        grids.append(grid^)
    
    var end = py_time.time()
    if total_runs > 0:
        var runs_float = py_builtins.float(total_runs)
        var avg_prep = py_operator.truediv(total_prep, runs_float)
        var avg_compute = py_operator.truediv(total_compute, runs_float)
        var avg_transfer = py_operator.truediv(total_transfer, runs_float)
        var avg_total = py_operator.truediv(total_total, runs_float)
        
        var avg_prep_str = py_builtins.format(avg_prep, ".3f")
        var avg_compute_str = py_builtins.format(avg_compute, ".3f")
        var avg_transfer_str = py_builtins.format(avg_transfer, ".3f")
        var avg_total_str = py_builtins.format(avg_total, ".3f")
        var total_gpu_str = py_builtins.format(total_total, ".3f")
        
        var elapsed = py_builtins.format(end - start, ".3f")
        print("Generated", len(rule_container.rules), "grids sequentially on CPU (parallel cells on CuPy GPU) in", elapsed, "seconds (", total_gpu_str, "s on CuPy GPU)", sep=" ")
        
        print("Average CuPy GPU timings across", total_runs, "runs:",
              "prep", avg_prep_str, "s | compute", avg_compute_str, "s | transfer", avg_transfer_str, "s | total", avg_total_str, "s")
    
    return grids^

fn main() raises:
    var py_time = Python.import_module("time")
    var py_builtins = Python.import_module("builtins")
    
    var total_start = py_time.time()
    var rule_container = RuleContainer()
    
    _ = generate_sequential_grids_cpu_sequential_cells_cpu(rule_container)
    var grids_seq_cpu_cells_par_cpu = generate_sequential_grids_cpu_parallel_cells_cpu(rule_container)
    _ = generate_parallel_grids_cpu_sequential_cells_cpu(rule_container)
    _ = generate_sequential_grids_cpu_parallel_cells_gpu(rule_container)
    
    if RENDER_PNGS:
        var renderer = Renderer()
        renderer.save_pngs(grids_seq_cpu_cells_par_cpu, rule_container)
    else:
        print("Skipping PNG generation for this run (benchmark mode)")
    
    var total_end = py_time.time()
    var total_elapsed = py_builtins.format(total_end - total_start, ".3f")
    print("Total time:", total_elapsed, "seconds")
