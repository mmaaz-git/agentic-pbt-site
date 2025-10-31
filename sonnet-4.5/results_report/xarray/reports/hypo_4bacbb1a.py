import numpy as np
from hypothesis import given, strategies as st, settings
from xarray.testing import assert_duckarray_equal


@given(
    x=st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100),
    y=st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100)
)
@settings(max_examples=500)
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
            pass  # Expected behavior for different values
        except TypeError as e:
            # Any TypeError when comparing arrays is a bug
            raise AssertionError(
                f"Bug: assert_duckarray_equal crashes on 0-d arrays. Error: {e}"
            )


if __name__ == "__main__":
    import traceback
    import sys

    print("Running Hypothesis test for 0-dimensional array handling...")
    print("=" * 60)

    # Run with a specific failing example first
    try:
        test_assert_duckarray_equal_0d_arrays.hypothesis.inner_test(x=5.0, y=3.0)
        print("Test passed with x=5.0, y=3.0 (unexpected!)")
    except AssertionError as e:
        print(f"AssertionError (bug found): {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"Other error: {e}")
        traceback.print_exc()

    # Run the full property test
    try:
        test_assert_duckarray_equal_0d_arrays()
        print("\nAll property tests passed!")
    except Exception as e:
        print(f"\nProperty test failed: {e}")
        traceback.print_exc()
        sys.exit(1)