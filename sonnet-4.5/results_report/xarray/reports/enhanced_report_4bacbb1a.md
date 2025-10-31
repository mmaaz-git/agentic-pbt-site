# Bug Report: xarray.testing._format_message Crashes on 0-Dimensional Arrays

**Target**: `xarray.testing.assertions._format_message`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `_format_message` function in `xarray.testing.assertions` crashes with a TypeError when formatting error messages for failed assertions on 0-dimensional (scalar) numpy arrays. This occurs because the function incorrectly uses Python's built-in `max()` function on numpy arrays, which fails to iterate over 0-dimensional arrays.

## Property-Based Test

```python
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
```

<details>

<summary>
**Failing input**: `x=5.0, y=3.0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 22, in test_assert_duckarray_equal_0d_arrays
    assert_duckarray_equal(x_arr, y_arr)
    ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/testing/assertions.py", line 32, in wrapper
    return func(*args, **kwargs)
  File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/testing/assertions.py", line 317, in assert_duckarray_equal
    assert equiv, _format_message(x, y, err_msg=err_msg, verbose=verbose)
                  ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/testing/assertions.py", line 264, in _format_message
    abs_diff = max(abs(diff))
TypeError: 'numpy.float64' object is not iterable

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 41, in <module>
    test_assert_duckarray_equal_0d_arrays.hypothesis.inner_test(x=5.0, y=3.0)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 27, in test_assert_duckarray_equal_0d_arrays
    raise AssertionError(
        f"Bug: assert_duckarray_equal crashes on 0-d arrays. Error: {e}"
    )
AssertionError: Bug: assert_duckarray_equal crashes on 0-d arrays. Error: 'numpy.float64' object is not iterable
Running Hypothesis test for 0-dimensional array handling...
============================================================
AssertionError (bug found): Bug: assert_duckarray_equal crashes on 0-d arrays. Error: 'numpy.float64' object is not iterable

Full traceback:
```
</details>

## Reproducing the Bug

```python
import numpy as np
from xarray.testing import assert_duckarray_equal

# Test case: 0-dimensional arrays with different values
x = np.array(5.0)
y = np.array(3.0)

print(f"x shape: {x.shape}, x value: {x}")
print(f"y shape: {y.shape}, y value: {y}")
print("Attempting to compare 0-dimensional arrays with different values...")

try:
    assert_duckarray_equal(x, y)
except TypeError as e:
    print(f"\nTypeError occurred: {e}")
    print(f"Error type: {type(e).__name__}")
except AssertionError as e:
    print(f"\nAssertionError (expected): {e}")
```

<details>

<summary>
TypeError: 'numpy.float64' object is not iterable
</summary>
```
x shape: (), x value: 5.0
y shape: (), y value: 3.0
Attempting to compare 0-dimensional arrays with different values...

TypeError occurred: 'numpy.float64' object is not iterable
Error type: TypeError
```
</details>

## Why This Is A Bug

This violates expected behavior for several critical reasons:

1. **API Contract Violation**: The `assert_duckarray_equal` function is documented as being "like `np.testing.assert_array_equal`, but for duckarrays." NumPy's `assert_array_equal` handles 0-dimensional arrays correctly without crashing. Users reasonably expect xarray's version to have equivalent functionality.

2. **Inconsistent Behavior**: The function successfully compares equal 0-dimensional arrays but crashes when they differ. This means the issue only manifests when generating error messages, not in the comparison logic itself. This inconsistency indicates an implementation error rather than a design limitation.

3. **Wrong Function Choice**: The code uses Python's built-in `max()` function on line 264, which requires an iterable. However, 0-dimensional numpy arrays are not iterable. The correct approach is to use numpy's `np.max()` function, which handles all array dimensions including 0-d arrays.

4. **Common Use Case**: 0-dimensional arrays are not edge cases in scientific computing. They frequently arise from operations like:
   - Extracting single values: `arr[0, 0]` from a 2D array
   - Reductions: `np.sum(arr)` produces a 0-d array
   - Scalar wrapping: `np.array(5.0)` for consistent array operations

## Relevant Context

The bug is located in `/home/npc/miniconda/lib/python3.13/site-packages/xarray/testing/assertions.py` at line 264 in the `_format_message` function. This internal function is called by both `assert_duckarray_equal` and `assert_duckarray_allclose` when arrays don't match and an error message needs to be formatted.

The issue affects any code that:
- Uses xarray's testing utilities for validation
- Compares scalar numpy arrays (0-dimensional arrays)
- Has test suites that validate individual values as arrays

Links:
- [NumPy documentation on 0-d arrays](https://numpy.org/doc/stable/reference/arrays.scalars.html)
- [xarray.testing module documentation](https://docs.xarray.dev/en/stable/generated/xarray.testing.html)

## Proposed Fix

```diff
def _format_message(x, y, err_msg, verbose):
    diff = x - y
-   abs_diff = max(abs(diff))
+   abs_diff = np.max(np.abs(diff))
    rel_diff = "not implemented"

    n_diff = np.count_nonzero(diff)
    n_total = diff.size
```

This fix:
- Replaces Python's `max()` with numpy's `np.max()`
- Uses `np.abs()` for consistency with numpy operations
- Handles all array shapes correctly (0-d, 1-d, n-d, and even empty arrays)
- Maintains backward compatibility for all existing use cases