# Bug Report: pandas.core.dtypes.common.ensure_python_int raises OverflowError instead of TypeError

**Target**: `pandas.core.dtypes.common.ensure_python_int`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `ensure_python_int` function raises `OverflowError` when given infinity values, but its docstring promises to raise `TypeError` for values that can't be converted to int.

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
```

**Failing input**: `float('inf')` and `float('-inf')`

## Reproducing the Bug

```python
from pandas.core.dtypes.common import ensure_python_int

ensure_python_int(float('inf'))
```

This raises:
```
OverflowError: cannot convert float infinity to integer
```

Expected: Should raise `TypeError` as documented in the docstring.

## Why This Is A Bug

The function's docstring states:

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

The docstring promises that the function raises `TypeError` if the value "can't be converted" to an int. Infinity clearly cannot be converted to an int, so according to the documented contract, it should raise `TypeError`.

However, the implementation catches `TypeError`, `ValueError`, and `AssertionError` but not `OverflowError`:

```python
try:
    new_value = int(value)
    assert new_value == value
except (TypeError, ValueError, AssertionError) as err:
    raise TypeError(f"Wrong type {type(value)} for value {value}") from err
```

When `int(float('inf'))` is called, it raises `OverflowError`, which is not caught and re-raised as `TypeError`.

## Fix

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