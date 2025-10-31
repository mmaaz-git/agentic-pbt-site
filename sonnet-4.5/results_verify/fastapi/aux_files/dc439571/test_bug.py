#!/usr/bin/env python3
"""Reproduce the purported bug in fastapi.dependencies.utils.get_path_param_names"""

# First, let's verify the bug exists
from fastapi.dependencies.utils import get_path_param_names

# Test the reproduction case
path_with_space = "/{ }"
path_with_tab = "/{\t}"
path_with_newline = "/{\n}"

print("Testing get_path_param_names with different whitespace:")
print(f"Space: {get_path_param_names(path_with_space)}")
print(f"Tab: {get_path_param_names(path_with_tab)}")
print(f"Newline: {get_path_param_names(path_with_newline)}")
print()

# Run the hypothesis test
from hypothesis import given, strategies as st

@given(st.lists(st.text(alphabet=st.characters(blacklist_characters='{}'), min_size=1)))
def test_get_path_param_names_extracts_params(param_names):
    path = '/' + '/'.join(['{' + name + '}' for name in param_names])
    result = get_path_param_names(path)
    assert result == set(param_names), f"Failed for {param_names}: got {result}"

print("Running property-based test...")
try:
    test_get_path_param_names_extracts_params()
    print("Property test passed!")
except AssertionError as e:
    print(f"Property test failed: {e}")
except Exception as e:
    print(f"Error during property test: {e}")

# Test specifically with newline
print("\nTesting specific failing case ['\n']:")
param_names = ['\n']
path = '/' + '/'.join(['{' + name + '}' for name in param_names])
result = get_path_param_names(path)
print(f"Input: {param_names!r}")
print(f"Path: {path!r}")
print(f"Result: {result}")
print(f"Expected: {set(param_names)}")
print(f"Match: {result == set(param_names)}")