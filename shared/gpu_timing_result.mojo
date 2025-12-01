struct GPUTimingResult(Copyable, Movable):
    var prep: Float64
    var compute: Float64
    var transfer: Float64
    var total: Float64
    var runs: Int
    
    fn __init__(out self, prep: Float64, compute: Float64, transfer: Float64, total: Float64, runs: Int):
        self.prep = prep
        self.compute = compute
        self.transfer = transfer
        self.total = total
        self.runs = runs

