#!/usr/bin/env python3
"""Test what the expected behavior should be for (dict, [])"""

# Test what dict([]) actually does
print("Testing what dict([]) returns:")
result = dict([])
print(f"dict([]) = {result}")
print(f"Type: {type(result)}")

# Verify this is what we expect
assert result == {}
print("✓ dict([]) returns an empty dictionary")

# Let's also test what other operations do
print("\nTesting other similar operations:")
print(f"list([]) = {list([])}")
print(f"tuple([]) = {tuple([])}")
print(f"set([]) = {set([])}")

# Check if (dict, []) is indeed a valid task
from dask.core import istask
expr = (dict, [])
print(f"\nIs (dict, []) a valid dask task? {istask(expr)}")

# Test similar expressions that might work
from dask.diagnostics.profile_visualize import unquote

print("\nTesting similar expressions that work:")

# Test dict with non-empty list
try:
    expr2 = (dict, [[["key", "value"]]])
    result2 = unquote(expr2)
    print(f"unquote((dict, [[['key', 'value']]])) = {result2}")
except Exception as e:
    print(f"Failed: {e}")

# Test empty list
try:
    expr3 = (list, [[]])
    result3 = unquote(expr3)
    print(f"unquote((list, [[]])) = {result3}")
except Exception as e:
    print(f"Failed: {e}")

# Test empty tuple
try:
    expr4 = (tuple, [[]])
    result4 = unquote(expr4)
    print(f"unquote((tuple, [[]])) = {result4}")
except Exception as e:
    print(f"Failed: {e}")