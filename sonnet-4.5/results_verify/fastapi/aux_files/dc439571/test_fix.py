#!/usr/bin/env python3
"""Test the proposed fix for the bug"""

import re

# Current implementation
def get_path_param_names_current(path: str):
    return set(re.findall("{(.*?)}", path))

# Proposed fix
def get_path_param_names_fixed(path: str):
    return set(re.findall("{([^}]*)}", path))

# Test cases
test_paths = [
    "/{ }",        # space
    "/{\t}",       # tab
    "/{\n}",       # newline
    "/{foo\nbar}", # newline in middle
    "/{a}/{b}",    # normal case
    "/{}",         # empty
    "/{_test}",    # underscore
    "/{123}",      # numbers
]

print("Comparing current vs fixed implementation:\n")
print("Path                  | Current              | Fixed")
print("-" * 60)
for path in test_paths:
    current = get_path_param_names_current(path)
    fixed = get_path_param_names_fixed(path)
    path_repr = repr(path).ljust(20)
    current_repr = str(current).ljust(20)
    fixed_repr = str(fixed).ljust(20)
    print(f"{path_repr} | {current_repr} | {fixed_repr}")

# Test with Python regex dotall flag as alternative
def get_path_param_names_dotall(path: str):
    return set(re.findall("{(.*?)}", path, re.DOTALL))

print("\n\nAlternative with re.DOTALL flag:")
print("Path                  | With DOTALL")
print("-" * 40)
for path in test_paths:
    result = get_path_param_names_dotall(path)
    path_repr = repr(path).ljust(20)
    result_repr = str(result).ljust(20)
    print(f"{path_repr} | {result_repr}")