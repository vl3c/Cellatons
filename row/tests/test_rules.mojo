"""Unit tests for Rule application.

Run with: mojo test row/tests/
"""

from testing import assert_true, assert_equal, assert_false
from row.rule import Rule


# ─────────────────────────────────────────────────────────────────────────────
# Rule 30 Tests
# ─────────────────────────────────────────────────────────────────────────────

fn create_rule_30() -> Rule:
    """Helper to create Rule 30."""
    var groups = List[List[String]]()
    var g1 = List[String]()
    g1.append("100")
    g1.append("011")
    groups.append(g1^)
    var g2 = List[String]()
    g2.append("010")
    g2.append("001")
    groups.append(g2^)
    return Rule("Rule 30", "rule_30", groups^)


fn test_rule30_100_produces_1() raises:
    """Test Rule 30: pattern 100 -> 1."""
    var rule = create_rule_30()
    assert_equal(rule.apply(1, 0, 0), 1)


fn test_rule30_011_produces_1() raises:
    """Test Rule 30: pattern 011 -> 1."""
    var rule = create_rule_30()
    assert_equal(rule.apply(0, 1, 1), 1)


fn test_rule30_010_produces_1() raises:
    """Test Rule 30: pattern 010 -> 1."""
    var rule = create_rule_30()
    assert_equal(rule.apply(0, 1, 0), 1)


fn test_rule30_001_produces_1() raises:
    """Test Rule 30: pattern 001 -> 1."""
    var rule = create_rule_30()
    assert_equal(rule.apply(0, 0, 1), 1)


fn test_rule30_000_produces_0() raises:
    """Test Rule 30: pattern 000 -> 0."""
    var rule = create_rule_30()
    assert_equal(rule.apply(0, 0, 0), 0)


fn test_rule30_111_produces_0() raises:
    """Test Rule 30: pattern 111 -> 0."""
    var rule = create_rule_30()
    assert_equal(rule.apply(1, 1, 1), 0)


fn test_rule30_110_produces_0() raises:
    """Test Rule 30: pattern 110 -> 0."""
    var rule = create_rule_30()
    assert_equal(rule.apply(1, 1, 0), 0)


fn test_rule30_101_produces_0() raises:
    """Test Rule 30: pattern 101 -> 0."""
    var rule = create_rule_30()
    assert_equal(rule.apply(1, 0, 1), 0)


# ─────────────────────────────────────────────────────────────────────────────
# Rule 110 Tests
# ─────────────────────────────────────────────────────────────────────────────

fn create_rule_110() -> Rule:
    """Helper to create Rule 110."""
    var groups = List[List[String]]()
    var g1 = List[String]()
    g1.append("110")
    g1.append("101")
    groups.append(g1^)
    var g2 = List[String]()
    g2.append("011")
    g2.append("010")
    groups.append(g2^)
    var g3 = List[String]()
    g3.append("001")
    groups.append(g3^)
    return Rule("Rule 110", "rule_110", groups^)


fn test_rule110_110_produces_1() raises:
    """Test Rule 110: pattern 110 -> 1."""
    var rule = create_rule_110()
    assert_equal(rule.apply(1, 1, 0), 1)


fn test_rule110_101_produces_1() raises:
    """Test Rule 110: pattern 101 -> 1."""
    var rule = create_rule_110()
    assert_equal(rule.apply(1, 0, 1), 1)


fn test_rule110_011_produces_1() raises:
    """Test Rule 110: pattern 011 -> 1."""
    var rule = create_rule_110()
    assert_equal(rule.apply(0, 1, 1), 1)


fn test_rule110_010_produces_1() raises:
    """Test Rule 110: pattern 010 -> 1."""
    var rule = create_rule_110()
    assert_equal(rule.apply(0, 1, 0), 1)


fn test_rule110_001_produces_1() raises:
    """Test Rule 110: pattern 001 -> 1."""
    var rule = create_rule_110()
    assert_equal(rule.apply(0, 0, 1), 1)


fn test_rule110_000_produces_0() raises:
    """Test Rule 110: pattern 000 -> 0."""
    var rule = create_rule_110()
    assert_equal(rule.apply(0, 0, 0), 0)


fn test_rule110_111_produces_0() raises:
    """Test Rule 110: pattern 111 -> 0."""
    var rule = create_rule_110()
    assert_equal(rule.apply(1, 1, 1), 0)


fn test_rule110_100_produces_0() raises:
    """Test Rule 110: pattern 100 -> 0."""
    var rule = create_rule_110()
    assert_equal(rule.apply(1, 0, 0), 0)

