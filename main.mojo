from python import Python
from rule import Rule
from grid import Grid
from rule_container import RuleContainer
from algorithm import parallelize

alias WIDTH: Int = 2000
alias HEIGHT: Int = 1000

fn get_filename(output_name: String) -> String:
    var filename = String("generated/")
    filename += output_name
    filename += ".png"
    return filename^

fn main() raises:
    var py_time = Python.import_module("time")
    var py_builtins = Python.import_module("builtins")
    
    var total_start = py_time.time()
    
    var rule_container = RuleContainer()
    
    for rule in rule_container.rules:
        var rule_start = py_time.time()
        
        var grid = Grid(WIDTH, HEIGHT)
        grid.generate(rule)
        
        var filename = get_filename(rule.output_name)
        
        grid.save_png(filename)
        
        var rule_end = py_time.time()
        var rule_elapsed = py_builtins.format(rule_end - rule_start, ".3f")
        print(rule.name, "saved to", filename, "in", rule_elapsed, "seconds")
    
    var total_end = py_time.time()
    var total_elapsed = py_builtins.format(total_end - total_start, ".3f")
    print("Generated", len(rule_container.rules), "rules in", total_elapsed, "seconds")

