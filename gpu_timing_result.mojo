from python import PythonObject

struct GPUTimingResult(Copyable, Movable):
    var prep: PythonObject
    var compute: PythonObject
    var transfer: PythonObject
    var total: PythonObject
    var runs: Int
    
    fn __init__(out self, prep: PythonObject, compute: PythonObject, transfer: PythonObject, total: PythonObject, runs: Int):
        self.prep = prep
        self.compute = compute
        self.transfer = transfer
        self.total = total
        self.runs = runs

