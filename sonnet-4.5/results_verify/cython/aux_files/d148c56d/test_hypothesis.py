#!/usr/bin/env python3
import string
from hypothesis import given, strategies as st, settings

@given(st.text(alphabet=string.ascii_letters + '_', min_size=1, max_size=10).filter(str.isidentifier))
@settings(max_examples=10)  # Reduced for testing
def test_fill_command_py_prefix_strips_prefix(var_name):
    arg_string = f"py:{var_name}"

    name = arg_string
    if name.startswith('py:'):
        parsed_name = name[:3]  # The buggy code

    expected_name = var_name
    actual_name = parsed_name

    try:
        assert actual_name == expected_name, f"Variable name should be {expected_name!r}, got {actual_name!r}"
        print(f"PASS: {var_name}")
    except AssertionError as e:
        print(f"FAIL: {e}")
        return False
    return True

# Run the test
test_fill_command_py_prefix_strips_prefix()