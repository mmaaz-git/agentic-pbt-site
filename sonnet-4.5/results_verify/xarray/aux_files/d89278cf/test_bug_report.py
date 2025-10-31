import numpy as np
from hypothesis import given, strategies as st, settings
from xarray.testing import assert_duckarray_equal

# First, let's run the simple reproduction case
print("Testing simple reproduction case:")
x = np.array(5.0)
y = np.array(3.0)

print(f"x shape: {x.shape}, x value: {x}")
print(f"y shape: {y.shape}, y value: {y}")
print(f"x is 0-d array: {x.ndim == 0}")
print(f"y is 0-d array: {y.ndim == 0}")

try:
    assert_duckarray_equal(x, y)
    print("ERROR: Assert should have failed but didn't")
except TypeError as e:
    print(f"TypeError occurred as reported: {e}")
except AssertionError as e:
    print(f"AssertionError occurred (expected behavior): {e}")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")

print("\n" + "="*60 + "\n")

# Now let's run the property-based test
@given(
    x=st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100),
    y=st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100)
)
@settings(max_examples=50)  # Reduced for testing
def test_assert_duckarray_equal_0d_arrays(x, y):
    """
    assert_duckarray_equal should handle 0-dimensional arrays gracefully.
    """
    x_arr = np.array(x)
    y_arr = np.array(y)

    if x == y:
        assert_duckarray_equal(x_arr, y_arr)
    else:
        try:
            assert_duckarray_equal(x_arr, y_arr)
        except AssertionError:
            pass  # This is expected
        except TypeError as e:
            if "iteration over a 0-d array" in str(e) or "0-d" in str(e):
                raise AssertionError(
                    f"Bug: assert_duckarray_equal crashes on 0-d arrays. Error: {e}"
                )

print("Running property-based test...")
try:
    test_assert_duckarray_equal_0d_arrays()
    print("Property test passed - no issues found")
except AssertionError as e:
    print(f"Property test found bug: {e}")
except Exception as e:
    print(f"Unexpected error in property test: {type(e).__name__}: {e}")