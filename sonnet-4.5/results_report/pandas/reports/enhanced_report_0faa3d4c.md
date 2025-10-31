# Bug Report: pandas_dtype Raises ValueError Instead of TypeError for Invalid Dictionary Inputs

**Target**: `pandas.api.types.pandas_dtype`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `pandas_dtype` function violates its documented contract by raising `ValueError` instead of `TypeError` for certain invalid dictionary inputs, breaking the API promise stated in its docstring.

## Property-Based Test

```python
from pandas.api.types import pandas_dtype
from hypothesis import given, strategies as st
import pytest


@given(st.dictionaries(
    st.text(min_size=1, max_size=10),
    st.one_of(st.integers(), st.text()),
    min_size=1,
    max_size=10
))
def test_pandas_dtype_raises_only_typeerror(d):
    """pandas_dtype should only raise TypeError for invalid inputs, per docstring."""
    try:
        pandas_dtype(d)
    except TypeError:
        pass
    except ValueError as e:
        pytest.fail(
            f"pandas_dtype raised ValueError instead of TypeError for input {d!r}.\n"
            f"Docstring says it should only raise TypeError.\n"
            f"Error was: {e}"
        )

if __name__ == "__main__":
    test_pandas_dtype_raises_only_typeerror()
```

<details>

<summary>
**Failing input**: `{'0': ''}`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 15, in test_pandas_dtype_raises_only_typeerror
    pandas_dtype(d)
    ~~~~~~~~~~~~^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/common.py", line 1663, in pandas_dtype
    npdtype = np.dtype(dtype)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/_internal.py", line 66, in _usefields
    names, formats, offsets, titles = _makenames_list(adict, align)
                                      ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/_internal.py", line 36, in _makenames_list
    raise ValueError("entry not a 2- or 3- tuple")
ValueError: entry not a 2- or 3- tuple

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 26, in <module>
    test_pandas_dtype_raises_only_typeerror()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 7, in test_pandas_dtype_raises_only_typeerror
    st.text(min_size=1, max_size=10),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 19, in test_pandas_dtype_raises_only_typeerror
    pytest.fail(
    ~~~~~~~~~~~^
        f"pandas_dtype raised ValueError instead of TypeError for input {d!r}.\n"
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        f"Docstring says it should only raise TypeError.\n"
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        f"Error was: {e}"
        ^^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/outcomes.py", line 177, in fail
    raise Failed(msg=reason, pytrace=pytrace)
Failed: pandas_dtype raised ValueError instead of TypeError for input {'0': ''}.
Docstring says it should only raise TypeError.
Error was: entry not a 2- or 3- tuple
Falsifying example: test_pandas_dtype_raises_only_typeerror(
    d={'0': ''},
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/28/hypo.py:18
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/_internal.py:35
```
</details>

## Reproducing the Bug

```python
from pandas.api.types import pandas_dtype

invalid_input = {'0': ''}

try:
    pandas_dtype(invalid_input)
except ValueError as e:
    print(f"BUG: Raised ValueError: {e}")
    print("Expected: TypeError (per docstring)")
except TypeError as e:
    print(f"OK: Raised TypeError as documented: {e}")
```

<details>

<summary>
Output shows ValueError raised instead of expected TypeError
</summary>
```
BUG: Raised ValueError: entry not a 2- or 3- tuple
Expected: TypeError (per docstring)
```
</details>

## Why This Is A Bug

The `pandas_dtype` function's docstring explicitly promises to raise only `TypeError` for invalid inputs:

```python
def pandas_dtype(dtype) -> DtypeObj:
    """
    ...
    Raises
    ------
    TypeError if not a dtype
    """
```

However, when passed invalid dictionary inputs (which NumPy interprets as structured dtype specifications), the function allows `ValueError` exceptions from `np.dtype()` to propagate unchanged. This violates the documented API contract.

The implementation at lines 1663-1666 in `/pandas/core/dtypes/common.py` only catches `SyntaxError` and converts it to `TypeError`, but misses `ValueError` which `np.dtype()` also raises for malformed dictionary inputs:

- Dictionary entries with non-tuple values (e.g., `{'0': ''}`) trigger `ValueError: entry not a 2- or 3- tuple`
- Dictionary entries with wrong tuple sizes also trigger `ValueError`
- The existing comment "np.dtype uses `eval` which can raise SyntaxError" shows the intent to standardize exceptions but is incomplete

## Relevant Context

This bug affects multiple invalid dictionary patterns:
- `{'0': ''}`: ValueError - "entry not a 2- or 3- tuple"
- `{'b': 'text'}`: ValueError - "entry not a 2- or 3- tuple"
- `{'key': [1, 2]}`: ValueError - "entry not a 2- or 3- tuple"
- `{'x': (1,)}`: ValueError - "entry not a 2- or 3- tuple" (wrong tuple size)
- `{'a': 1}`: TypeError - "object of type 'int' has no len()" (correctly raises TypeError)

NumPy uses dictionaries to define structured dtypes with field specifications. Valid entries should be 2- or 3-tuples like `{'field': (np.int32, 4)}`. Invalid entries cause NumPy to raise `ValueError` from its internal validation in `numpy/_core/_internal.py`.

Documentation: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.api.types.pandas_dtype.html

## Proposed Fix

```diff
--- a/pandas/core/dtypes/common.py
+++ b/pandas/core/dtypes/common.py
@@ -1661,8 +1661,8 @@ def pandas_dtype(dtype) -> DtypeObj:
             # Hence enabling DeprecationWarning
             warnings.simplefilter("always", DeprecationWarning)
             npdtype = np.dtype(dtype)
-    except SyntaxError as err:
-        # np.dtype uses `eval` which can raise SyntaxError
+    except (SyntaxError, ValueError) as err:
+        # np.dtype uses `eval` which can raise SyntaxError, and raises ValueError for invalid dict entries
         raise TypeError(f"data type '{dtype}' not understood") from err

     # Any invalid dtype (such as pd.Timestamp) should raise an error.
```