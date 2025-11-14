from python import Python
from rule import Rule
from algorithm import parallelize

alias CELL_SIZE: Int = 5

struct Grid(Copyable, Movable):
    var cells: List[List[Int]]
    var width: Int
    var height: Int
    
    fn __init__(out self, width: Int, height: Int):
        self.width = width
        self.height = height
        self.cells = List[List[Int]]()
        
        for _ in range(height):
            var row = List[Int]()
            for _ in range(width):
                row.append(0)
            self.cells.append(row^)
    
    fn __copyinit__(out self, existing: Self):
        self.width = existing.width
        self.height = existing.height
        self.cells = List[List[Int]]()
        for i in range(len(existing.cells)):
            var row = List[Int]()
            for j in range(len(existing.cells[i])):
                row.append(existing.cells[i][j])
            self.cells.append(row^)
    
    fn __moveinit__(out self, deinit existing: Self):
        self.width = existing.width
        self.height = existing.height
        self.cells = existing.cells^
    
    fn set_cell(mut self, row: Int, col: Int, value: Int):
        self.cells[row][col] = value
    
    fn get_cell(self, row: Int, col: Int) -> Int:
        return self.cells[row][col]
    
    fn generate_parallel_cpu(mut self, rule: Rule):
        self.cells[0][self.width // 2] = 1
        
        for row in range(1, self.height):
            @parameter
            fn compute_cell(col: Int):
                if col >= 1 and col < self.width - 1:
                    var left = self.cells[row - 1][col - 1]
                    var center = self.cells[row - 1][col]
                    var right = self.cells[row - 1][col + 1]
                    
                    self.cells[row][col] = rule.apply(left, center, right)
            
            parallelize[compute_cell](self.width)
    
    fn generate_sequential_cpu(mut self, rule: Rule):
        self.cells[0][self.width // 2] = 1
        
        for row in range(1, self.height):
            for col in range(1, self.width - 1):
                var left = self.cells[row - 1][col - 1]
                var center = self.cells[row - 1][col]
                var right = self.cells[row - 1][col + 1]
                
                self.cells[row][col] = rule.apply(left, center, right)
    
    fn print_to_console(self):
        for row in range(self.height):
            var line = String("")
            for col in range(self.width):
                if self.cells[row][col] == 1:
                    line += "O"
                else:
                    line += " "
            print(line)
    
    fn save_png(self, filename: String) raises:
        var PIL = Python.import_module("PIL.Image")
        var builtins = Python.import_module("builtins")
        
        var img_width = self.width * CELL_SIZE
        var img_height = self.height * CELL_SIZE
        
        var size = builtins.tuple([img_width, img_height])
        var img = PIL.new("RGB", size, "white")
        var pixels = img.load()
        
        for row in range(self.height):
            for col in range(self.width):
                if self.cells[row][col] == 1:
                    for py in range(CELL_SIZE):
                        for px in range(CELL_SIZE):
                            var x = col * CELL_SIZE + px
                            var y = row * CELL_SIZE + py
                            var black = builtins.tuple([0, 0, 0])
                            pixels[x, y] = black
        
        img.save(filename)

