# Bug Report: pandas.core.dtypes.common.ensure_python_int Raises OverflowError Instead of Documented TypeError for Infinity Values

**Target**: `pandas.core.dtypes.common.ensure_python_int`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `ensure_python_int` function raises `OverflowError` when given infinity values (float('inf') or float('-inf')), contradicting its docstring which promises to raise `TypeError` for values that can't be converted to int.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.core.dtypes.common import ensure_python_int

@given(st.one_of(st.just(float('inf')), st.just(float('-inf'))))
@settings(max_examples=20)
def test_ensure_python_int_infinity_raises_typeerror(value):
    try:
        result = ensure_python_int(value)
        assert False, f"Should have raised TypeError, got {result}"
    except TypeError:
        pass
    except Exception as e:
        assert False, f"Expected TypeError but got {type(e).__name__}: {e}"

if __name__ == "__main__":
    test_ensure_python_int_infinity_raises_typeerror()
```

<details>

<summary>
**Failing input**: `value=inf`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 8, in test_ensure_python_int_infinity_raises_typeerror
    result = ensure_python_int(value)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/common.py", line 116, in ensure_python_int
    new_value = int(value)
OverflowError: cannot convert float infinity to integer

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 16, in <module>
    test_ensure_python_int_infinity_raises_typeerror()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 5, in test_ensure_python_int_infinity_raises_typeerror
    @settings(max_examples=20)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/21/hypo.py", line 13, in test_ensure_python_int_infinity_raises_typeerror
    assert False, f"Expected TypeError but got {type(e).__name__}: {e}"
           ^^^^^
AssertionError: Expected TypeError but got OverflowError: cannot convert float infinity to integer
Falsifying example: test_ensure_python_int_infinity_raises_typeerror(
    value=inf,
)
```
</details>

## Reproducing the Bug

```python
from pandas.core.dtypes.common import ensure_python_int

# Test with positive infinity
print("Testing with float('inf'):")
try:
    result = ensure_python_int(float('inf'))
    print(f"Result: {result}")
except Exception as e:
    print(f"{type(e).__name__}: {e}")

print("\nTesting with float('-inf'):")
# Test with negative infinity
try:
    result = ensure_python_int(float('-inf'))
    print(f"Result: {result}")
except Exception as e:
    print(f"{type(e).__name__}: {e}")
```

<details>

<summary>
Raises OverflowError for both positive and negative infinity
</summary>
```
Testing with float('inf'):
OverflowError: cannot convert float infinity to integer

Testing with float('-inf'):
OverflowError: cannot convert float infinity to integer
```
</details>

## Why This Is A Bug

The function's docstring explicitly states that it raises `TypeError` "if the value isn't an int or can't be converted to one":

```python
def ensure_python_int(value: int | np.integer) -> int:
    """
    Ensure that a value is a python int.

    ...

    Raises
    ------
    TypeError: if the value isn't an int or can't be converted to one.
    """
```

Infinity mathematically cannot be converted to an integer, so according to the documented contract, the function should raise `TypeError`. However, the implementation at lines 115-119 only catches `TypeError`, `ValueError`, and `AssertionError`, but not `OverflowError`:

```python
try:
    new_value = int(value)
    assert new_value == value
except (TypeError, ValueError, AssertionError) as err:
    raise TypeError(f"Wrong type {type(value)} for value {value}") from err
```

When Python's built-in `int()` function is called with infinity, it raises `OverflowError`, which is not caught and thus propagates directly to the caller, violating the documented API contract.

## Relevant Context

The function already demonstrates clear intent to normalize all conversion failures to `TypeError` by catching and re-raising other exceptions (`TypeError`, `ValueError`, `AssertionError`). The omission of `OverflowError` appears to be an oversight rather than intentional design.

This is an internal utility function in pandas located at `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/common.py`. The issue affects pandas users who rely on exception handling and expect the documented `TypeError` when passing non-convertible values to this function.

Python's standard behavior for `int(float('inf'))` is to raise `OverflowError`, which is why this exception needs to be caught and converted to `TypeError` to maintain the documented contract.

## Proposed Fix

```diff
def ensure_python_int(value: int | np.integer) -> int:
    """
    Ensure that a value is a python int.

    Parameters
    ----------
    value: int or numpy.integer

    Returns
    -------
    int

    Raises
    ------
    TypeError: if the value isn't an int or can't be converted to one.
    """
    if not (is_integer(value) or is_float(value)):
        if not is_scalar(value):
            raise TypeError(
                f"Value needs to be a scalar value, was type {type(value).__name__}"
            )
        raise TypeError(f"Wrong type {type(value)} for value {value}")
    try:
        new_value = int(value)
        assert new_value == value
-   except (TypeError, ValueError, AssertionError) as err:
+   except (TypeError, ValueError, AssertionError, OverflowError) as err:
        raise TypeError(f"Wrong type {type(value)} for value {value}") from err
    return new_value
```