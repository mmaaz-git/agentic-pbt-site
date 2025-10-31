# Bug Report: scipy.integrate Functions Raise Cryptic IndexError for Invalid Axis Values

**Target**: `scipy.integrate.trapezoid`, `scipy.integrate.simpson`, `scipy.integrate.cumulative_trapezoid`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

When an invalid `axis` parameter is provided to `trapezoid`, `simpson`, or `cumulative_trapezoid`, these functions raise cryptic `IndexError` messages that give no indication of the actual problem, forcing users to guess what went wrong or dive into scipy's source code.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, settings, strategies as st
from scipy import integrate
import pytest


@given(
    st.integers(min_value=2, max_value=5),
    st.integers(min_value=2, max_value=5)
)
@settings(max_examples=50)
def test_trapezoid_invalid_axis_error_message(dim1, dim2):
    """Test that trapezoid raises meaningful error for invalid axis."""
    y = np.ones((dim1, dim2))
    invalid_axis = y.ndim + 1  # axis=3 for 2D array

    with pytest.raises((ValueError, Exception)) as exc_info:
        integrate.trapezoid(y, axis=invalid_axis)

    # Check that error message mentions 'axis' or 'bound' to be informative
    assert 'axis' in str(exc_info.value).lower() or 'bound' in str(exc_info.value).lower(), \
        f"Error message should mention 'axis' or 'bound', got: {exc_info.value}"


@given(
    st.integers(min_value=2, max_value=5),
    st.integers(min_value=2, max_value=5)
)
@settings(max_examples=50)
def test_simpson_invalid_axis_error_message(dim1, dim2):
    """Test that simpson raises meaningful error for invalid axis."""
    y = np.ones((dim1, dim2))
    invalid_axis = y.ndim + 1  # axis=3 for 2D array

    with pytest.raises((ValueError, Exception)) as exc_info:
        integrate.simpson(y, axis=invalid_axis)

    # Check that error message mentions 'axis' or 'bound' to be informative
    assert 'axis' in str(exc_info.value).lower() or 'bound' in str(exc_info.value).lower(), \
        f"Error message should mention 'axis' or 'bound', got: {exc_info.value}"


@given(
    st.integers(min_value=2, max_value=5),
    st.integers(min_value=2, max_value=5)
)
@settings(max_examples=50)
def test_cumulative_trapezoid_invalid_axis_error_message(dim1, dim2):
    """Test that cumulative_trapezoid raises meaningful error for invalid axis."""
    y = np.ones((dim1, dim2))
    invalid_axis = y.ndim + 1  # axis=3 for 2D array

    with pytest.raises((ValueError, Exception)) as exc_info:
        integrate.cumulative_trapezoid(y, axis=invalid_axis)

    # Check that error message mentions 'axis' or 'bound' to be informative
    assert 'axis' in str(exc_info.value).lower() or 'bound' in str(exc_info.value).lower(), \
        f"Error message should mention 'axis' or 'bound', got: {exc_info.value}"


if __name__ == "__main__":
    # Run the tests with pytest
    pytest.main([__file__, "-v"])
```

<details>

<summary>
**Failing input**: `dim1=2, dim2=2` (or any dimensions with invalid axis)
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/5
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 3 items

hypo.py::test_trapezoid_invalid_axis_error_message FAILED                [ 33%]
hypo.py::test_simpson_invalid_axis_error_message FAILED                  [ 66%]
hypo.py::test_cumulative_trapezoid_invalid_axis_error_message FAILED     [100%]

=================================== FAILURES ===================================
__________________ test_trapezoid_invalid_axis_error_message ___________________

    @given(
>       st.integers(min_value=2, max_value=5),
                   ^^^
        st.integers(min_value=2, max_value=5)
    )

hypo.py:8:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

dim1 = 2, dim2 = 2

    @given(
        st.integers(min_value=2, max_value=5),
        st.integers(min_value=2, max_value=5)
    )
    @settings(max_examples=50)
    def test_trapezoid_invalid_axis_error_message(dim1, dim2):
        """Test that trapezoid raises meaningful error for invalid axis."""
        y = np.ones((dim1, dim2))
        invalid_axis = y.ndim + 1  # axis=3 for 2D array

        with pytest.raises((ValueError, Exception)) as exc_info:
            integrate.trapezoid(y, axis=invalid_axis)

        # Check that error message mentions 'axis' or 'bound' to be informative
>       assert 'axis' in str(exc_info.value).lower() or 'bound' in str(exc_info.value).lower(), \
            f"Error message should mention 'axis' or 'bound', got: {exc_info.value}"
E       AssertionError: Error message should mention 'axis' or 'bound', got: list assignment index out of range
E       assert ('axis' in 'list assignment index out of range' or 'bound' in 'list assignment index out of range')
E        +  where 'list assignment index out of range' = <built-in method lower of str object at 0x756449062f60>()
E        +    where <built-in method lower of str object at 0x756449062f60> = 'list assignment index out of range'.lower
E        +      where 'list assignment index out of range' = str(IndexError('list assignment index out of range'))
E        +        where IndexError('list assignment index out of range') = <ExceptionInfo IndexError('list assignment index out of range') tblen=2>.value
E        +  and   'list assignment index out of range' = <built-in method lower of str object at 0x756449062f60>()
E        +    where <built-in method lower of str object at 0x756449062f60> = 'list assignment index out of range'.lower
E        +      where 'list assignment index out of range' = str(IndexError('list assignment index out of range'))
E        +        where IndexError('list assignment index out of range') = <ExceptionInfo IndexError('list assignment index out of range') tblen=2>.value
E       Falsifying example: test_trapezoid_invalid_axis_error_message(
E           dim1=2,  # or any other generated value
E           dim2=2,  # or any other generated value
E       )

hypo.py:21: AssertionError
___________________ test_simpson_invalid_axis_error_message ____________________

    @given(
>       st.integers(min_value=2, max_value=5),
                   ^^^
        st.integers(min_value=2, max_value=5)
    )

hypo.py:26:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

dim1 = 2, dim2 = 2

    @given(
        st.integers(min_value=2, max_value=5),
        st.integers(min_value=2, max_value=5)
    )
    @settings(max_examples=50)
    def test_simpson_invalid_axis_error_message(dim1, dim2):
        """Test that simpson raises meaningful error for invalid axis."""
        y = np.ones((dim1, dim2))
        invalid_axis = y.ndim + 1  # axis=3 for 2D array

        with pytest.raises((ValueError, Exception)) as exc_info:
            integrate.simpson(y, axis=invalid_axis)

        # Check that error message mentions 'axis' or 'bound' to be informative
>       assert 'axis' in str(exc_info.value).lower() or 'bound' in str(exc_info.value).lower(), \
            f"Error message should mention 'axis' or 'bound', got: {exc_info.value}"
E       AssertionError: Error message should mention 'axis' or 'bound', got: tuple index out of range
E       assert ('axis' in 'tuple index out of range' or 'bound' in 'tuple index out of range')
E        +  where 'tuple index out of range' = <built-in method lower of str object at 0x756448fbb000>()
E        +    where <built-in method lower of str object at 0x756448fbb000> = 'tuple index out of range'.lower
E        +      where 'tuple index out of range' = str(IndexError('tuple index out of range'))
E        +        where IndexError('tuple index out of range') = <ExceptionInfo IndexError('tuple index out of range') tblen=2>.value
E        +  and   'tuple index out of range' = <built-in method lower of str object at 0x756448fbb000>()
E        +    where <built-in method lower of str object at 0x756448fbb000> = 'tuple index out of range'.lower
E        +      where 'tuple index out of range' = str(IndexError('tuple index out of range'))
E        +        where IndexError('tuple index out of range') = <ExceptionInfo IndexError('tuple index out of range') tblen=2>.value
E       Falsifying example: test_simpson_invalid_axis_error_message(
E           dim1=2,  # or any other generated value
E           dim2=2,  # or any other generated value
E       )

hypo.py:39: AssertionError
_____________ test_cumulative_trapezoid_invalid_axis_error_message _____________

    @given(
>       st.integers(min_value=2, max_value=5),
                   ^^^
        st.integers(min_value=2, max_value=5)
    )

hypo.py:44:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

dim1 = 2, dim2 = 2

    @given(
        st.integers(min_value=2, max_value=5),
        st.integers(min_value=2, max_value=5)
    )
    @settings(max_examples=50)
    def test_cumulative_trapezoid_invalid_axis_error_message(dim1, dim2):
        """Test that cumulative_trapezoid raises meaningful error for invalid axis."""
        y = np.ones((dim1, dim2))
        invalid_axis = y.ndim + 1  # axis=3 for 2D array

        with pytest.raises((ValueError, Exception)) as exc_info:
            integrate.cumulative_trapezoid(y, axis=invalid_axis)

        # Check that error message mentions 'axis' or 'bound' to be informative
>       assert 'axis' in str(exc_info.value).lower() or 'bound' in str(exc_info.value).lower(), \
            f"Error message should mention 'axis' or 'bound', got: {exc_info.value}"
E       AssertionError: Error message should mention 'axis' or 'bound', got: tuple index out of range
E       assert ('axis' in 'tuple index out of range' or 'bound' in 'tuple index out of range')
E        +  where 'tuple index out of range' = <built-in method lower of str object at 0x756448e10210>()
E        +    where <built-in method lower of str object at 0x756448e10210> = 'tuple index out of range'.lower
E        +      where 'tuple index out of range' = str(IndexError('tuple index out of range'))
E        +        where IndexError('tuple index out of range') = <ExceptionInfo IndexError('tuple index out of range') tblen=2>.value
E        +  and   'tuple index out of range' = <built-in method lower of str object at 0x756448e10210>()
E        +    where <built-in method lower of str object at 0x756448e10210> = 'tuple index out of range'.lower
E        +      where 'tuple index out of range' = str(IndexError('tuple index out of range'))
E        +        where IndexError('tuple index out of range') = <ExceptionInfo IndexError('tuple index out of range') tblen=2>.value
E       Falsifying example: test_cumulative_trapezoid_invalid_axis_error_message(
E           dim1=2,  # or any other generated value
E           dim2=2,  # or any other generated value
E       )

hypo.py:57: AssertionError
=============================== warnings summary ===============================
../../../../miniconda/lib/python3.13/site-packages/_pytest/config/__init__.py:1290
  /home/npc/miniconda/lib/python3.13/site-packages/_pytest/config/__init__.py:1290: PytestAssertRewriteWarning: Module already imported so cannot be rewritten; _hypothesis_globals
    self._mark_plugins_for_rewrite(hook, disable_autoload)

../../../../miniconda/lib/python3.13/site-packages/_pytest/config/__init__.py:1290
  /home/npc/miniconda/lib/python3.13/site-packages/_pytest/config/__init__.py:1290: PytestAssertRewriteWarning: Module already imported so cannot be rewritten; hypothesis
    self._mark_plugins_for_rewrite(hook, disable_autoload)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED hypo.py::test_trapezoid_invalid_axis_error_message - AssertionError: E...
FAILED hypo.py::test_simpson_invalid_axis_error_message - AssertionError: Err...
FAILED hypo.py::test_cumulative_trapezoid_invalid_axis_error_message - Assert...
======================== 3 failed, 2 warnings in 0.25s =========================
```
</details>

## Reproducing the Bug

```python
import numpy as np
from scipy import integrate

# Create a simple 2D array
y = np.array([[1, 2, 3],
              [4, 5, 6]])

print("Testing with 2D array shape:", y.shape)
print("Testing with invalid axis=2 (out of bounds for 2D array)")
print()

# Test trapezoid with invalid axis
print("1. Testing integrate.trapezoid(y, axis=2):")
try:
    result = integrate.trapezoid(y, axis=2)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
print()

# Test simpson with invalid axis
print("2. Testing integrate.simpson(y, axis=2):")
try:
    result = integrate.simpson(y, axis=2)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
print()

# Test cumulative_trapezoid with invalid axis
print("3. Testing integrate.cumulative_trapezoid(y, axis=2):")
try:
    result = integrate.cumulative_trapezoid(y, axis=2)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
print()

# For comparison, test cumulative_simpson which handles this correctly
print("4. Testing integrate.cumulative_simpson(y, axis=2) for comparison:")
try:
    result = integrate.cumulative_simpson(y, axis=2)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
print()

# For comparison, test numpy.sum which also handles this correctly
print("5. Testing numpy.sum(y, axis=2) for comparison:")
try:
    result = np.sum(y, axis=2)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
```

<details>

<summary>
Output showing cryptic errors for the first three functions and clear errors for comparison functions
</summary>
```
Testing with 2D array shape: (2, 3)
Testing with invalid axis=2 (out of bounds for 2D array)

1. Testing integrate.trapezoid(y, axis=2):
Error type: IndexError
Error message: list assignment index out of range

2. Testing integrate.simpson(y, axis=2):
Error type: IndexError
Error message: tuple index out of range

3. Testing integrate.cumulative_trapezoid(y, axis=2):
Error type: IndexError
Error message: tuple index out of range

4. Testing integrate.cumulative_simpson(y, axis=2) for comparison:
Error type: ValueError
Error message: `axis=2` is not valid for `y` with `y.ndim=2`.

5. Testing numpy.sum(y, axis=2) for comparison:
Error type: AxisError
Error message: axis 2 is out of bounds for array of dimension 2
```
</details>

## Why This Is A Bug

This violates the principle of clear error communication in library design. When users provide an invalid axis parameter, they receive cryptic IndexError messages like "list assignment index out of range" or "tuple index out of range" that provide no indication that the axis parameter is the problem. This forces users to either:

1. Guess what went wrong based on the cryptic message
2. Dive into scipy's source code to understand where the error occurs
3. Use trial and error to figure out what parameter caused the issue

The bug is particularly egregious because:
- **Inconsistency within the same module**: `cumulative_simpson` in the same scipy.integrate module already handles this correctly with a clear error message
- **Documentation doesn't specify error behavior**: None of the functions document what errors should be raised for invalid axis values
- **Standard practice violation**: NumPy raises clear `AxisError` messages, establishing user expectations
- **The errors occur at different internal points**: Each function fails at a different line when trying to use the invalid axis value directly

## Relevant Context

The issue stems from these functions not validating the axis parameter before using it:

- In `trapezoid` (line 131): `slice1[axis] = slice(1, None)` - directly indexes a list with invalid axis
- In `simpson` (line 445): `N = y.shape[axis]` - directly indexes shape tuple with invalid axis
- In `cumulative_trapezoid` (line 303): `if y.shape[axis] == 0:` - directly indexes shape tuple with invalid axis

Meanwhile, `cumulative_simpson` (lines 738-744) properly handles this by catching the numpy AxisError and re-raising with a clear message.

Relevant scipy documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.trapezoid.html

## Proposed Fix

```diff
--- a/scipy/integrate/_quadrature.py
+++ b/scipy/integrate/_quadrature.py
@@ -125,6 +125,10 @@ def trapezoid(y, x=None, dx=1.0, axis=-1):
     # cf. https://github.com/scipy/scipy/pull/21524#issuecomment-2354105942
     result_dtype = xp_result_type(y, force_floating=True, xp=xp)
     nd = y.ndim
+    if axis >= nd or axis < -nd:
+        raise ValueError(
+            f"`axis={axis}` is not valid for `y` with `y.ndim={nd}`."
+        )
     slice1 = [slice(None)]*nd
     slice2 = [slice(None)]*nd
     slice1[axis] = slice(1, None)
@@ -300,6 +304,10 @@ def cumulative_trapezoid(y, x=None, dx=1.0, axis=-1, initial=None):

     """
     y = np.asarray(y)
+    if axis >= y.ndim or axis < -y.ndim:
+        raise ValueError(
+            f"`axis={axis}` is not valid for `y` with `y.ndim={y.ndim}`."
+        )
     if y.shape[axis] == 0:
         raise ValueError("At least one point is required along `axis`.")
     if x is None:
@@ -442,6 +450,10 @@ def simpson(y, x=None, *, dx=1.0, axis=-1):
     """
     y = np.asarray(y)
     nd = len(y.shape)
+    if axis >= nd or axis < -nd:
+        raise ValueError(
+            f"`axis={axis}` is not valid for `y` with `y.ndim={nd}`."
+        )
     N = y.shape[axis]
     last_dx = dx
     returnshape = 0
```