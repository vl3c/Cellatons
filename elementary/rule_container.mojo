from elementary.rule import Rule

struct RuleContainer:
    var rules: List[Rule]
    
    fn __init__(out self):
        self.rules = List[Rule]()
        
        var rule254_groups: List[List[String]] = [
            ["111", "110", "101"],
            ["100", "011", "010"],
            ["001"]
        ]
        self.rules.append(Rule("Rule 254", "rule_254", rule254_groups^))
        
        var rule30_groups: List[List[String]] = [
            ["100", "011"],
            ["010", "001"]
        ]
        self.rules.append(Rule("Rule 30", "rule_30", rule30_groups^))
        
        var rule110_groups: List[List[String]] = [
            ["110", "101"],
            ["011", "010"],
            ["001"]
        ]
        self.rules.append(Rule("Rule 110", "rule_110", rule110_groups^))

