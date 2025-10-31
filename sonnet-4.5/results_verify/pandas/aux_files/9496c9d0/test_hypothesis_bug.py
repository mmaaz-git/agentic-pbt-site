#!/usr/bin/env python3
"""Test the Hypothesis property test from the bug report"""

from hypothesis import given, strategies as st, assume
from fastapi.security.utils import get_authorization_scheme_param


@given(st.text(min_size=1), st.text(min_size=1), st.integers(min_value=2, max_value=10))
def test_get_authorization_scheme_param_multiple_spaces(scheme_input, param_input, num_spaces):
    assume(" " not in scheme_input)
    assume(" " not in param_input)

    authorization = f"{scheme_input}{' ' * num_spaces}{param_input}"

    scheme, param = get_authorization_scheme_param(authorization)

    assert scheme == scheme_input
    assert param == param_input, f"Expected {repr(param_input)}, got {repr(param)}"


if __name__ == "__main__":
    # Run the test with the specific failing input mentioned
    print("Running with specific failing input from bug report:")
    scheme_input = '0'
    param_input = '0'
    num_spaces = 2

    authorization = f"{scheme_input}{' ' * num_spaces}{param_input}"
    print(f"Authorization header: {repr(authorization)}")

    scheme, param = get_authorization_scheme_param(authorization)

    print(f"Scheme: {repr(scheme)}")
    print(f"Param: {repr(param)}")

    try:
        assert scheme == scheme_input
        print("✓ Scheme matches")
    except AssertionError:
        print(f"✗ Scheme mismatch: expected {repr(scheme_input)}, got {repr(scheme)}")

    try:
        assert param == param_input
        print("✓ Param matches")
    except AssertionError:
        print(f"✗ Param mismatch: expected {repr(param_input)}, got {repr(param)}")

    print("\nRunning full Hypothesis test...")
    try:
        test_get_authorization_scheme_param_multiple_spaces()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")