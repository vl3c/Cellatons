from python import Python, PythonObject
from row.rule import Rule
from row.grid import Grid
from row.rule_container import RuleContainer
from algorithm import parallelize
from shared.common import WIDTH, HEIGHT, RENDER_PNGS
from shared.benchmark import BenchmarkSuite
from shared.logger import Logger
from row.renderer import Renderer


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
    bench.add("seq grids / seq cells", py_time.time() - start1)
    
    # 2. CPU sequential/parallel
    var grids2 = List[Grid]()
    var start2 = py_time.time()
    for rule in rule_container.rules:
        var grid = Grid(WIDTH, HEIGHT, logger)
        grid.generate_parallel_cpu(rule)
        grids2.append(grid^)
    bench.add("seq grids / par cells", py_time.time() - start2)
    
    # 3. CPU parallel/sequential
    var grids3 = List[Grid]()
    for _ in range(num_rules):
        grids3.append(Grid(WIDTH, HEIGHT, logger))
    var start3 = py_time.time()
    @parameter
    fn gen3(i: Int):
        grids3[i].generate_sequential_cpu(rule_container.rules[i])
    parallelize[gen3](num_rules)
    bench.add("par grids / seq cells", py_time.time() - start3)
    
    # 4. CPU sequential/SIMD
    var grids4a = List[Grid]()
    var start4a = py_time.time()
    for rule in rule_container.rules:
        var grid = Grid(WIDTH, HEIGHT, logger)
        grid.generate_simd_cpu(rule)
        grids4a.append(grid^)
    bench.add("seq grids / SIMD cells", py_time.time() - start4a)
    
    # 5. CPU parallel/SIMD
    var grids5a = List[Grid]()
    for _ in range(num_rules):
        grids5a.append(Grid(WIDTH, HEIGHT, logger))
    var start5a = py_time.time()
    @parameter
    fn gen5a(i: Int):
        grids5a[i].generate_simd_cpu(rule_container.rules[i])
    parallelize[gen5a](num_rules)
    bench.add("par grids / SIMD cells", py_time.time() - start5a)
    
    # 6. CuPy GPU
    var grids6 = List[Grid]()
    var g6: Float64 = 0.0
    var start6 = py_time.time()
    for rule in rule_container.rules:
        var grid = Grid(WIDTH, HEIGHT, logger)
        var stats = grid.generate_parallel_cells_cupy_gpu(rule)
        g6 += stats.total
        grids6.append(grid^)
    bench.add_gpu("CuPy GPU", py_time.time() - start6, g6)
    
    # 7. Native GPU
    var grids7 = List[Grid]()
    var g7: Float64 = 0.0
    var start7 = py_time.time()
    for rule in rule_container.rules:
        var grid = Grid(WIDTH, HEIGHT, logger)
        var stats = grid.generate_native_gpu(rule)
        g7 += stats.total
        grids7.append(grid^)
    bench.add_gpu("Native GPU", py_time.time() - start7, g7)
    
    # 8. Native GPU (no CPU allocation)
    var g8: Float64 = 0.0
    var start8 = py_time.time()
    for rule in rule_container.rules:
        var stats = Grid.benchmark_native_gpu(rule, logger)
        g8 += stats.total
    bench.add_gpu("Native GPU (no CPU alloc)", py_time.time() - start8, g8)
    
    # 9. Native GPU (multi-row batching - reduced kernel launches)
    var g9: Float64 = 0.0
    var start9 = py_time.time()
    for rule in rule_container.rules:
        var stats = Grid.benchmark_native_gpu_multirow(rule, logger)
        g9 += stats.total
    bench.add_gpu("Native GPU (batched)", py_time.time() - start9, g9)
    
    # Render if enabled (using fastest method: par grids / SIMD cells)
    if RENDER_PNGS:
        var renderer = Renderer()
        renderer.save_pngs(grids5a, rule_container)
    
    # Print summary
    bench.print_summary()
