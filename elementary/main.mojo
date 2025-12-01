from python import Python, PythonObject
from elementary.rule import Rule
from elementary.grid import Grid
from elementary.rule_container import RuleContainer
from algorithm import parallelize
from shared.common import WIDTH, HEIGHT, RENDER_PNGS
from shared.benchmark import BenchmarkSuite
from shared.logger import Logger
from elementary.renderer import Renderer


fn main() raises:
    var logger = Logger()  # Create logger at start - all timestamps relative to this
    
    var py_time = Python.import_module("time")
    var py_builtins = Python.import_module("builtins")
    var py_operator = Python.import_module("operator")
    
    var rule_container = RuleContainer()
    var num_rules = len(rule_container.rules)
    var bench = BenchmarkSuite(num_rules)
    
    # 1. CPU sequential/sequential
    var grids1 = List[Grid]()
    var start1 = py_time.time()
    for rule in rule_container.rules:
        var grid = Grid(WIDTH, HEIGHT, logger)
        grid.generate_sequential_cpu(rule)
        grids1.append(grid^)
    bench.add("CPU seq/seq", py_time.time() - start1)
    
    # 2. CPU sequential/parallel
    var grids2 = List[Grid]()
    var start2 = py_time.time()
    for rule in rule_container.rules:
        var grid = Grid(WIDTH, HEIGHT, logger)
        grid.generate_parallel_cpu(rule)
        grids2.append(grid^)
    bench.add("CPU seq/par", py_time.time() - start2)
    
    # 3. CPU parallel/sequential
    var grids3 = List[Grid]()
    for _ in range(num_rules):
        grids3.append(Grid(WIDTH, HEIGHT, logger))
    var start3 = py_time.time()
    @parameter
    fn gen3(i: Int):
        grids3[i].generate_sequential_cpu(rule_container.rules[i])
    parallelize[gen3](num_rules)
    bench.add("CPU par/seq", py_time.time() - start3)
    
    # 4. CuPy GPU
    var grids4 = List[Grid]()
    var g4: Float64 = 0.0
    var start4 = py_time.time()
    for rule in rule_container.rules:
        var grid = Grid(WIDTH, HEIGHT, logger)
        var stats = grid.generate_parallel_cells_cupy_gpu(rule)
        g4 += stats.total
        grids4.append(grid^)
    bench.add_gpu("CuPy GPU", py_time.time() - start4, g4)
    
    # 5. Native GPU
    var grids5 = List[Grid]()
    var g5: Float64 = 0.0
    var start5 = py_time.time()
    for rule in rule_container.rules:
        var grid = Grid(WIDTH, HEIGHT, logger)
        var stats = grid.generate_native_gpu(rule)
        g5 += stats.total
        grids5.append(grid^)
    bench.add_gpu("Native GPU", py_time.time() - start5, g5)
    
    # 6. Native GPU (no CPU allocation)
    var g6: Float64 = 0.0
    var start6 = py_time.time()
    for rule in rule_container.rules:
        var stats = Grid.benchmark_native_gpu(rule, logger)
        g6 += stats.total
    bench.add_gpu("Native GPU (no CPU alloc)", py_time.time() - start6, g6)
    
    # Render if enabled
    if RENDER_PNGS:
        var renderer = Renderer()
        renderer.save_pngs(grids2, rule_container)
    
    # Print summary
    bench.print_summary()
