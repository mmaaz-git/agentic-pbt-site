#!/usr/bin/env python3
"""Test script to reproduce the WhereNode.relabel_aliases bug"""

# First, let's try to reproduce the bug exactly as described
from django.db.models.sql.where import WhereNode, AND, OR, XOR

print("Testing WhereNode.relabel_aliases return behavior")
print("=" * 50)

# Test 1: Empty change_map
print("\nTest 1: Empty change_map")
node1 = WhereNode(connector=AND, negated=False)
result_empty = node1.relabel_aliases({})
print(f"Empty map returns: {result_empty}")
print(f"Returns self: {result_empty is node1}")
print(f"Type of return: {type(result_empty)}")

# Test 2: Non-empty change_map
print("\nTest 2: Non-empty change_map")
node2 = WhereNode(connector=AND, negated=False)
result_nonempty = node2.relabel_aliases({'old_alias': 'new_alias'})
print(f"Non-empty map returns: {result_nonempty}")
print(f"Returns self: {result_nonempty is node2}")
print(f"Type of return: {type(result_nonempty)}")

# Test 3: Different connectors
print("\nTest 3: Testing with different connectors")
for connector in [AND, OR, XOR]:
    node = WhereNode(connector=connector, negated=False)
    result = node.relabel_aliases({'a': 'b'})
    print(f"{connector}: returns {result}, is self: {result is node}")

# Test 4: With negation
print("\nTest 4: Testing with negation")
node_neg = WhereNode(connector=AND, negated=True)
result_neg = node_neg.relabel_aliases({'x': 'y'})
print(f"Negated node: returns {result_neg}, is self: {result_neg is node_neg}")