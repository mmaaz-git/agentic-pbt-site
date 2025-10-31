#!/usr/bin/env python3
"""Minimal reproduction of the bug in llm.utils.remove_dict_none_values"""

from llm.utils import remove_dict_none_values

# Test the failing input reported
test_dict = {"a": [1, None, 3]}
result = remove_dict_none_values(test_dict)

print("Test 1: Simple list with None")
print(f"Input:  {test_dict}")
print(f"Output: {result}")
print(f"None still present: {None in result.get('a', [])}")
print()

# Test with nested structures
test_dict2 = {"a": [1, None, 3], "b": {"c": [None, 2]}}
result2 = remove_dict_none_values(test_dict2)

print("Test 2: Nested dict with lists containing None")
print(f"Input:  {test_dict2}")
print(f"Output: {result2}")
print(f"None in a: {None in result2.get('a', [])}")
print(f"None in b.c: {None in result2.get('b', {}).get('c', [])}")
print()

# Test with None as direct dict value
test_dict3 = {"a": None, "b": [None], "c": {"d": None, "e": [None]}}
result3 = remove_dict_none_values(test_dict3)

print("Test 3: Mix of None in dict values and lists")
print(f"Input:  {test_dict3}")
print(f"Output: {result3}")
print()

# Show what the docstring claims
print("Function docstring:")
print(f'"{remove_dict_none_values.__doc__.strip()}"')