#!/usr/bin/env python3
"""Test Python's string.Template behavior for comparison"""

from string import Template

print("=== Testing Python's string.Template ===")
template = Template("$x")
input_dict = {'x': 42}

print(f"Before: {input_dict}")
print(f"Keys before: {list(input_dict.keys())}")
result = template.substitute(input_dict)
print(f"After:  {input_dict}")
print(f"Keys after: {list(input_dict.keys())}")
print(f"Result: {result}")

if len(input_dict.keys()) > 1:
    print("\nPython's string.Template ALSO mutates the input!")
else:
    print("\nPython's string.Template does NOT mutate the input dict")