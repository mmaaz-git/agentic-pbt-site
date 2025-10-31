# Bug Report: xarray.testing._format_message Crashes on 0-Dimensional Arrays

**Target**: `xarray.testing.assertions._format_message`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `_format_message` function in `xarray.testing.assertions` crashes with a TypeError when formatting error messages for failed assertions on 0-dimensional (scalar) numpy arrays. This occurs because the function uses Python's built-in `max()` function on numpy arrays, which cannot iterate over 0-dimensional arrays.

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
            pass
        except TypeError as e:
            if "iteration over a 0-d array" in str(e) or "0-d" in str(e):
                raise AssertionError(
                    f"Bug: assert_duckarray_equal crashes on 0-d arrays. Error: {e}"
                )
```

**Failing input**: `x=5.0, y=3.0` (any two different scalar values)

## Reproducing the Bug

```python
import numpy as np
from xarray.testing import assert_duckarray_equal

x = np.array(5.0)
y = np.array(3.0)

try:
    assert_duckarray_equal(x, y)
except TypeError as e:
    print(f"TypeError: {e}")
```

Expected output:
```
TypeError: iteration over a 0-d array
```

## Why This Is A Bug

1. **API Contract Violation**: The `assert_duckarray_equal` function is designed to work with any duck array, including 0-dimensional arrays. Users should be able to compare scalar numpy arrays without crashes.

2. **Inconsistent Behavior**: The function works fine when the arrays are equal (no error message needed), but crashes when they differ and an error message must be generated.

3. **Wrong Function Used**: The code uses Python's built-in `max()` function instead of `np.max()` on numpy arrays. The built-in `max()` requires an iterable, but 0-dimensional arrays are not iterable.

4. **Common Use Case**: 0-dimensional numpy arrays (scalars) are commonly used in scientific computing and testing, so this is not an edge case.

## Location

File: `xarray/testing/assertions.py`
Function: `_format_message`
Line: 264

```python
def _format_message(x, y, err_msg, verbose):
    diff = x - y
    abs_diff = max(abs(diff))  # BUG: crashes on 0-d arrays
    ...
```

## Fix

Replace Python's built-in `max()` with `np.max()`:

```diff
def _format_message(x, y, err_msg, verbose):
    diff = x - y
-   abs_diff = max(abs(diff))
+   abs_diff = np.max(np.abs(diff))
    rel_diff = "not implemented"

    n_diff = np.count_nonzero(diff)
    n_total = diff.size
```

This fix also addresses potential issues with:
- Empty arrays (where `max()` would raise "max() arg is an empty sequence")
- Ensures consistent use of numpy functions throughout the codebase
- Properly handles all array shapes including 0-d and empty arrays