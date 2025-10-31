#!/usr/bin/env python3
"""Run the hypothesis test from the bug report"""

from hypothesis import given, strategies as st, assume, settings
from django.db.models.sql.where import WhereNode, AND, OR, XOR


@st.composite
def change_map_strategy(draw):
    num_mappings = draw(st.integers(min_value=1, max_value=5))
    change_map = {}
    for _ in range(num_mappings):
        old_alias = draw(st.text(min_size=1, max_size=10))
        new_alias = draw(st.text(min_size=1, max_size=10))
        change_map[old_alias] = new_alias
    return change_map


@given(st.sampled_from([AND, OR, XOR]), st.booleans(), change_map_strategy())
@settings(max_examples=10)
def test_wherenode_relabel_aliases_mutates_and_returns_self(connector, negated, change_map):
    node = WhereNode(connector=connector, negated=negated)
    result = node.relabel_aliases(change_map)
    assert result is node, f"relabel_aliases should return self for method chaining, but returned {result}"

# Run the test
print("Running hypothesis test...")
try:
    test_wherenode_relabel_aliases_mutates_and_returns_self()
    print("Test passed! (This shouldn't happen if the bug exists)")
except AssertionError as e:
    print(f"Test failed as expected: {e}")

# Also test the specific failing input mentioned
print("\nTesting specific failing input from bug report:")
connector = 'AND'
negated = False
change_map = {'old': 'new'}
node = WhereNode(connector=connector, negated=negated)
result = node.relabel_aliases(change_map)
print(f"connector={connector!r}, negated={negated}, change_map={change_map}")
print(f"Result: {result}")
print(f"Is self: {result is node}")