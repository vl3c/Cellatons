from python import Python

alias WIDTH: Int = 2000
alias HEIGHT: Int = 1000

fn get_filename(output_name: String) -> String:
    var filename = String("generated/")
    filename += output_name
    filename += ".png"
    return filename^

fn format_time(elapsed: PythonObject) -> String:
    var py_builtins = Python.import_module("builtins")
    var elapsed_str: String = py_builtins.format(elapsed, ".3f").__str__()
    return elapsed_str

