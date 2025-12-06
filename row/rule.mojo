struct Rule(Copyable, Movable):
    var name: String
    var output_name: String
    var pattern_groups: List[List[String]]
    var pattern_mask: Int  # 8-bit mask for O(1) lookup
    var lookup_table: SIMD[DType.uint8, 8]  # Pre-expanded for SIMD shuffle
    
    fn __init__(out self, var name: String, var output_name: String, var pattern_groups: List[List[String]]):
        self.name = name^
        self.output_name = output_name^
        self.pattern_groups = pattern_groups^
        # Initialize pattern_mask after all other fields are set
        self.pattern_mask = 0
        self.lookup_table = SIMD[DType.uint8, 8](0)
        self.pattern_mask = self._build_mask()
        self.lookup_table = self._build_lookup_table()
    
    fn __copyinit__(out self, existing: Self):
        self.name = existing.name
        self.output_name = existing.output_name
        self.pattern_mask = existing.pattern_mask
        self.lookup_table = existing.lookup_table
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
        self.pattern_mask = existing.pattern_mask
        self.lookup_table = existing.lookup_table
    
    fn _build_mask(self) -> Int:
        """Pre-compute bitmask from pattern groups for O(1) lookup."""
        var mask: Int = 0
        for group in range(len(self.pattern_groups)):
            for idx in range(len(self.pattern_groups[group])):
                var code = Self._pattern_to_code(self.pattern_groups[group][idx])
                mask |= (1 << code)
        return mask
    
    fn _build_lookup_table(self) -> SIMD[DType.uint8, 8]:
        """Expand bitmask into 8-element lookup table for SIMD shuffle."""
        var table = SIMD[DType.uint8, 8](0)
        for i in range(8):
            table[i] = UInt8((self.pattern_mask >> i) & 1)
        return table
    
    @staticmethod
    fn _pattern_to_code(pattern: String) -> Int:
        """Convert pattern string like '101' to integer code (0-7)."""
        var value: Int = 0
        for idx in range(len(pattern)):
            value = value << 1
            if pattern[idx] == "1":
                value += 1
        return value
    
    fn matches_pattern(self, key: String) -> Bool:
        """Check if pattern matches (kept for backwards compatibility)."""
        for group in range(len(self.pattern_groups)):
            for idx in range(len(self.pattern_groups[group])):
                if self.pattern_groups[group][idx] == key:
                    return True
        return False
    
    @always_inline
    fn apply(self, left: Int, center: Int, right: Int) -> Int:
        """O(1) lookup using pre-computed bitmask."""
        var code = left * 4 + center * 2 + right
        return (self.pattern_mask >> code) & 1

