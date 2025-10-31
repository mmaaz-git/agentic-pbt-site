#!/usr/bin/env python3
import string
from hypothesis import given, strategies as st, settings, example

@given(st.text(alphabet=string.ascii_letters + '_', min_size=1, max_size=10).filter(str.isidentifier))
@settings(max_examples=5)  # Reduced for testing
@example("x")
@example("my_var")
@example("test_variable")
def test_fill_command_py_prefix_strips_prefix(var_name):
    arg_string = f"py:{var_name}"

    name = arg_string
    if name.startswith('py:'):
        parsed_name = name[:3]  # The buggy code

    expected_name = var_name
    actual_name = parsed_name

    print(f"Testing variable: {var_name}")
    print(f"  Expected: {expected_name!r}")
    print(f"  Got: {actual_name!r}")

    # This will always fail because of the bug
    assert actual_name == expected_name, f"Variable name should be {expected_name!r}, got {actual_name!r}"

# Run the test
if __name__ == "__main__":
    try:
        test_fill_command_py_prefix_strips_prefix()
        print("All tests passed!")
    except AssertionError as e:
        print(f"\nTest failed as expected due to bug: {e}")