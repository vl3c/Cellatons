struct Rule(Copyable, Movable):
    var name: String
    var output_name: String
    var pattern_groups: List[List[String]]
    
    fn __init__(out self, var name: String, var output_name: String, var pattern_groups: List[List[String]]):
        self.name = name^
        self.output_name = output_name^
        self.pattern_groups = pattern_groups^
    
    fn __copyinit__(out self, existing: Self):
        self.name = existing.name
        self.output_name = existing.output_name
        self.pattern_groups = List[List[String]]()
        for i in range(len(existing.pattern_groups)):
            var inner_list = List[String]()
            for j in range(len(existing.pattern_groups[i])):
                inner_list.append(existing.pattern_groups[i][j])
            self.pattern_groups.append(inner_list^)
    
    fn __moveinit__(out self, deinit existing: Self):
        self.name = existing.name^
        self.output_name = existing.output_name^
        self.pattern_groups = existing.pattern_groups^
    
    fn matches_pattern(self, key: String) -> Bool:
        for group in range(len(self.pattern_groups)):
            for idx in range(len(self.pattern_groups[group])):
                if self.pattern_groups[group][idx] == key:
                    return True
        return False
    
    fn apply(self, left: Int, center: Int, right: Int) -> Int:
        var key = String("")
        
        if left == 1:
            key += "1"
        else:
            key += "0"
        
        if center == 1:
            key += "1"
        else:
            key += "0"
        
        if right == 1:
            key += "1"
        else:
            key += "0"
        
        if self.matches_pattern(key):
            return 1
        
        return 0

