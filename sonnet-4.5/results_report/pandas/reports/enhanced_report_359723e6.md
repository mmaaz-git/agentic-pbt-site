# Bug Report: pandas.core.dtypes.common.ensure_python_int raises OverflowError instead of TypeError for infinity

**Target**: `pandas.core.dtypes.common.ensure_python_int`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `ensure_python_int` function raises `OverflowError` when given infinity values instead of the documented `TypeError`, violating its API contract.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.core.dtypes.common import ensure_python_int
import pytest
import numpy as np


@given(st.floats(allow_nan=True, allow_infinity=True))
@settings(max_examples=200)
def test_ensure_python_int_special_floats(value):
    if np.isnan(value) or np.isinf(value):
        with pytest.raises(TypeError):
            ensure_python_int(value)
    elif value == int(value):
        result = ensure_python_int(value)
        assert result == int(value)


if __name__ == "__main__":
    test_ensure_python_int_special_floats()
```

<details>

<summary>
**Failing input**: `inf`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 19, in <module>
    test_ensure_python_int_special_floats()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 8, in test_ensure_python_int_special_floats
    @settings(max_examples=200)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 12, in test_ensure_python_int_special_floats
    ensure_python_int(value)
    ~~~~~~~~~~~~~~~~~^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/common.py", line 116, in ensure_python_int
    new_value = int(value)
OverflowError: cannot convert float infinity to integer
Falsifying example: test_ensure_python_int_special_floats(
    value=inf,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/44/hypo.py:11
```
</details>

## Reproducing the Bug

```python
from pandas.core.dtypes.common import ensure_python_int

# Test with float infinity - this should raise TypeError according to docs
# but actually raises OverflowError
ensure_python_int(float('inf'))
```

<details>

<summary>
Raises OverflowError instead of documented TypeError
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/44/repo.py", line 5, in <module>
    ensure_python_int(float('inf'))
    ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/common.py", line 116, in ensure_python_int
    new_value = int(value)
OverflowError: cannot convert float infinity to integer
```
</details>

## Why This Is A Bug

The function's docstring at lines 93-107 in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/common.py` explicitly promises:

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

The documentation states that only `TypeError` will be raised for values that can't be converted to int. However, the implementation at lines 115-119:

```python
try:
    new_value = int(value)
    assert new_value == value
except (TypeError, ValueError, AssertionError) as err:
    raise TypeError(f"Wrong type {type(value)} for value {value}") from err
```

Only catches `TypeError`, `ValueError`, and `AssertionError`. When Python's built-in `int()` function receives `float('inf')`, it raises `OverflowError: cannot convert float infinity to integer`, which is not caught by the exception handler. This uncaught exception propagates to the caller, violating the documented contract.

The function already demonstrates intent to normalize all conversion failures to `TypeError` - it catches `ValueError` (raised by `int(float('nan'))`) and re-raises as `TypeError`. The same pattern should apply to `OverflowError`.

## Relevant Context

The function is part of pandas' public API (listed in `__all__` at line 1715). It performs type conversion validation, ensuring values can be safely converted to Python integers. The function correctly handles NaN values by catching `ValueError` and re-raising as `TypeError`, maintaining API consistency. However, infinity values slip through due to the missing `OverflowError` handler.

This inconsistency could cause issues in calling code that expects only `TypeError` based on the documentation. While the severity is low (infinity values are somewhat edge cases and the workaround is simple), it's still a legitimate bug that breaks the documented contract.

The pandas codebase shows that this function is used internally in various places where integer conversion validation is needed. Code relying on the documented behavior might fail unexpectedly when encountering infinity values.

## Proposed Fix

```diff
diff --git a/pandas/core/dtypes/common.py b/pandas/core/dtypes/common.py
index 1234567..abcdefg 100644
--- a/pandas/core/dtypes/common.py
+++ b/pandas/core/dtypes/common.py
@@ -115,7 +115,7 @@ def ensure_python_int(value: int | np.integer) -> int:
     try:
         new_value = int(value)
         assert new_value == value
-    except (TypeError, ValueError, AssertionError) as err:
+    except (TypeError, ValueError, AssertionError, OverflowError) as err:
         raise TypeError(f"Wrong type {type(value)} for value {value}") from err
     return new_value
```