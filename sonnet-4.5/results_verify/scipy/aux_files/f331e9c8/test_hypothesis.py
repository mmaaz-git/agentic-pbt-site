import numpy as np
from hypothesis import given, settings, strategies as st
from scipy import integrate
import pytest
import sys


@given(
    st.integers(min_value=2, max_value=5),
    st.integers(min_value=2, max_value=5)
)
@settings(max_examples=50)
def test_trapezoid_invalid_axis_error_message(dim1, dim2):
    y = np.ones((dim1, dim2))
    invalid_axis = y.ndim + 1

    with pytest.raises((ValueError, Exception)) as exc_info:
        integrate.trapezoid(y, axis=invalid_axis)

    assert 'axis' in str(exc_info.value).lower() or 'bound' in str(exc_info.value).lower(), \
        f"Error message should mention 'axis' or 'bound', got: {exc_info.value}"


@given(
    st.integers(min_value=2, max_value=5),
    st.integers(min_value=2, max_value=5)
)
@settings(max_examples=50)
def test_simpson_invalid_axis_error_message(dim1, dim2):
    y = np.ones((dim1, dim2))
    invalid_axis = y.ndim + 1

    with pytest.raises((ValueError, Exception)) as exc_info:
        integrate.simpson(y, axis=invalid_axis)

    assert 'axis' in str(exc_info.value).lower() or 'bound' in str(exc_info.value).lower(), \
        f"Error message should mention 'axis' or 'bound', got: {exc_info.value}"


@given(
    st.integers(min_value=2, max_value=5),
    st.integers(min_value=2, max_value=5)
)
@settings(max_examples=50)
def test_cumulative_trapezoid_invalid_axis_error_message(dim1, dim2):
    y = np.ones((dim1, dim2))
    invalid_axis = y.ndim + 1

    with pytest.raises((ValueError, Exception)) as exc_info:
        integrate.cumulative_trapezoid(y, axis=invalid_axis)

    assert 'axis' in str(exc_info.value).lower() or 'bound' in str(exc_info.value).lower(), \
        f"Error message should mention 'axis' or 'bound', got: {exc_info.value}"


if __name__ == "__main__":
    # Run the tests
    print("Running property-based tests...")
    print("=" * 60)

    print("\n1. Testing trapezoid error messages:")
    try:
        test_trapezoid_invalid_axis_error_message()
        print("✓ Test passed (error messages are informative)")
    except AssertionError as e:
        print(f"✗ Test failed: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

    print("\n2. Testing simpson error messages:")
    try:
        test_simpson_invalid_axis_error_message()
        print("✓ Test passed (error messages are informative)")
    except AssertionError as e:
        print(f"✗ Test failed: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

    print("\n3. Testing cumulative_trapezoid error messages:")
    try:
        test_cumulative_trapezoid_invalid_axis_error_message()
        print("✓ Test passed (error messages are informative)")
    except AssertionError as e:
        print(f"✗ Test failed: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")